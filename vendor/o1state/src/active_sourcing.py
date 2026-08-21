#!/usr/bin/env python3 -u
"""
ACTIVE SOURCING — the model curates its own stream.
===================================================
The living-stream flag showed a GSSM can TRAIN on an unbounded stream at O(1) memory. This
asks the next question: must we feed the model (passive schedule), or can the model FETCH its
own data (active sourcing)?

This is the neural translation of a proven, peer-reviewed pattern — the gap-driven discovery
loop from the IRI knowledge engine (Foss, IRI 2026): the system follows EPISTEMIC STRUCTURE,
not a fixed plan. There the gap was a convergence hub in a causal graph; here the gap is the
model's own SURPRISE — train on the source where you are most ignorant. Same loop shape
(gap → fetch → integrate), neural substrate. No symbolic apparatus is imported; the corpus is
just an iterator, and the policy chooses which iterator to pull next.

Three modes, one harness:
  passive-fixed   — round-robin a fixed schedule of sources (the baseline feeding)
  passive-random  — pick the next source uniformly at random (the control)
  active          — measure per-source surprise, pull from the highest-surprise source
                    (the model fetches its own cookies — gap-driven, neural)

The claim it tests: active sourcing reaches a given multi-source held-out loss in FEWER tokens
than a fixed schedule — the model curating its own stream beats being fed. Constant memory
throughout (it's the same O(1) streaming-train loop, only the next-block policy changes).
"""
import os, sys, json, argparse, time, threading, signal
sys.path.insert(0, "reference"); sys.path.insert(0, "src")

import resource
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    def _rss_gb(): return _PROC.memory_info().rss / 1e9
except ImportError:
    def _rss_gb():
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e9 if r > 1e7 else r / 1e6

import torch
import torch.nn as nn
torch.backends.mps.is_available = lambda: False
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

from streaming_train import StreamingNoPELM
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize


# ── the three sources, each a lazy block iterator over a different distribution ──────────────
SOURCES = {
    "c4":   lambda: ("allenai/c4", "en", "train"),
    "wiki": lambda: ("wikimedia/wikipedia", "20231101.en", "train"),
    "book": lambda: ("bookcorpus/bookcorpus", None, "train"),
}


def source_block_stream(name, stoi, unk, block=32768):
    """Lazily yield token blocks from one HF streaming source. The corpus is an iterator —
    swapping `name` swaps the whole distribution with zero change downstream."""
    from datasets import load_dataset
    spec = SOURCES[name]()
    if spec[1] is None:
        ds = load_dataset(spec[0], split=spec[2], streaming=True)
    else:
        ds = load_dataset(spec[0], spec[1], split=spec[2], streaming=True)
    pending = []
    for row in ds:
        t = row.get("text", "") if isinstance(row, dict) else ""
        if not t.strip():
            continue
        pending.extend(tokenize(t, stoi, unk))
        while len(pending) >= block:
            out, pending = pending[:block], pending[block:]
            yield out
    if pending:
        yield pending


class MultiSource:
    """Holds one persistent iterator per source (instantiate ONCE — never re-load in the hot
    loop, or HF Arrow buffers leak). `.pull(name, n)` returns ~n tokens from that source."""
    def __init__(self, names, stoi, unk, block=32768):
        self.names = names
        self.stoi, self.unk, self.block = stoi, unk, block
        self.iters = {n: source_block_stream(n, stoi, unk, block) for n in names}
        self.buf = {n: [] for n in names}

    def pull(self, name, n):
        b = self.buf[name]
        while len(b) < n:
            try:
                b.extend(next(self.iters[name]))
            except StopIteration:
                self.iters[name] = source_block_stream(name, self.stoi, self.unk, self.block)
                b.extend(next(self.iters[name]))
        out, self.buf[name] = b[:n], b[n:]
        return out


def _start_watchdog(out, hard_gb):
    def _watch():
        while True:
            if _rss_gb() > hard_gb:
                open(out + ".WATCHDOG_KILL", "w").write(f"rss={_rss_gb():.2f}GB\n")
                os.kill(os.getpid(), signal.SIGKILL)
            time.sleep(0.5)
    threading.Thread(target=_watch, daemon=True).start()


@torch.no_grad()
def source_surprise(model, ms, name, stoi, mask, dev, probe_tok=256):
    """The 'gap' signal: the model's current loss (surprise) on a small fresh probe from
    `name`. High surprise = the model is ignorant of this source right now = pull from it.
    Peeks at the source's buffer without consuming the training stream."""
    model.eval()
    ids = ms.buf[name][:probe_tok]
    if len(ids) < probe_tok:                       # top up the peek buffer if short
        ids = ms.pull(name, probe_tok)
        ms.buf[name] = ids + ms.buf[name]          # put it back — probe is non-consuming
    seg = ids[:probe_tok]
    x = torch.tensor(seg[:-1], dtype=torch.long).unsqueeze(0).to(dev)
    y = torch.tensor(seg[1:], dtype=torch.long).unsqueeze(0).to(dev)
    logits, _ = model(x, None)
    loss = float(nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)))
    model.train()
    return loss


@torch.no_grad()
def heldout_multi(model, heldout, chunk, dev, max_tokens=8000):
    """Mean held-out loss across ALL sources (the fair metric: a model that curates its own
    stream must still be good on every source, not just the one it liked)."""
    model.eval()
    lossf = nn.CrossEntropyLoss()
    per = {}
    for name, ids in heldout.items():
        tot, n, pos = 0.0, 0, 0
        ids = ids[:max_tokens]
        while pos + 1 < len(ids):
            seg = ids[pos:pos + chunk + 1]
            if len(seg) < 2:
                break
            x = torch.tensor(seg[:-1], dtype=torch.long).unsqueeze(0).to(dev)
            y = torch.tensor(seg[1:], dtype=torch.long).unsqueeze(0).to(dev)
            logits, _ = model(x, None)
            tot += float(lossf(logits.reshape(-1, logits.size(-1)), y.reshape(-1))) * (len(seg) - 1)
            n += len(seg) - 1
            pos += chunk
        per[name] = tot / max(1, n)
    model.train()
    return sum(per.values()) / len(per), per


def run(args):
    dev = torch.device("cpu")
    _start_watchdog(args.out, args.mem_hard_gb)
    names = args.sources.split(",")
    print(f"[safety] watchdog {args.mem_hard_gb}GB; ACTIVE-SOURCING mode={args.mode} "
          f"sources={names} (constant memory, the model picks its own stream)")

    # vocab from WT-2 (shared, deterministic); held-out = a fresh probe per source
    train_text, _ = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    V = len(vocab)
    ms = MultiSource(names, stoi, unk, block=args.block)
    # build a fixed held-out set per source BEFORE training (never trained on)
    heldout = {n: ms.pull(n, 8000) for n in names}
    print(f"  vocab={V}, held-out 8k tok/source built (never streamed for training)")

    torch.manual_seed(args.seed)
    model = StreamingNoPELM(V, mask, d_model=args.d_model, n_layers=2, n_heads=4,
                            d_head=args.d_model // 4, seq_len=32, dropout=0.0, causal=True).to(dev)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss()

    B, K = args.batch, args.chunk
    states = None
    n_tok = 0
    peak = _rss_gb()
    curve = []
    pulls = {n: 0 for n in names}
    rr = 0                                          # round-robin pointer for passive-fixed
    last_src = names[0]                             # active-mode: cached choice between probes
    prev_surp = None                                # active-lp: previous per-source surprise
    recent_losses = []                              # curate: rolling window of block losses
    seen_tok = 0                                    # curate: tokens looked at (incl. skipped)
    skipped = 0                                     # curate: blocks skipped
    replay_deck = []                                # curate-replay: external bounded "attic" of hard blocks
    replayed = 0                                    # curate-replay: count of re-practice steps
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    step = 0

    base, _ = heldout_multi(model, heldout, K, dev)
    print(f"  step 0: mean held-out {base:.3f}")
    while n_tok < args.target_tokens:
        # ── THE POLICY: which source feeds the next batch? ──
        if args.mode == "passive-fixed":
            src = names[rr % len(names)]; rr += 1
        elif args.mode == "passive-random":
            src = names[int(torch.randint(0, len(names), (1,), generator=gen))]
        elif args.mode == "active":
            # gap-driven v1 (naive): pull from the HIGHEST-SURPRISE source. KNOWN to lose to a
            # fixed schedule — highest surprise ≠ highest learning gain (curiosity collapse /
            # noisy-TV: the model chases the hardest source and starves the others). Kept as the
            # honest negative baseline against active-lp.
            if step % args.probe_every == 0:
                surp = {n: source_surprise(model, ms, n, stoi, mask, dev) for n in names}
                src = max(names, key=lambda n: surp[n])
                last_src = src
            else:
                src = last_src
        elif args.mode == "active-lp":
            # gap-driven v2 (LEARNING PROGRESS): pull from the source whose loss has FALLEN most
            # since we last measured it — train where training is WORKING, not where it's hardest.
            # This is the honest neural analog of the IRI gap-priority (structural importance /
            # convergence), not raw difficulty. Softmax over per-source learning progress so a
            # source is never fully starved (keeps the schedule balanced, beating the v1 collapse).
            if step % args.probe_every == 0:
                cur = {n: source_surprise(model, ms, n, stoi, mask, dev) for n in names}
                if prev_surp is not None:
                    lp = {n: prev_surp[n] - cur[n] for n in names}   # positive = improving
                    mx = max(lp.values())
                    w = {n: float(torch.exp(torch.tensor((lp[n] - mx) / max(1e-3, args.lp_temp)))) for n in names}
                    tot = sum(w.values())
                    r = float(torch.rand(1, generator=gen)) * tot
                    acc = 0.0
                    src = names[-1]
                    for n in names:
                        acc += w[n]
                        if r <= acc:
                            src = n; break
                    last_src = src
                else:
                    src = names[step % len(names)]
                prev_surp = cur
            else:
                src = last_src
        elif args.mode == "curate":
            src = names[rr % len(names)]; rr += 1   # every source still OFFERS blocks round-robin
        elif args.mode == "curate-replay":
            src = names[rr % len(names)]; rr += 1
        else:
            raise ValueError(args.mode)
        pulls[src] += 1

        toks = ms.pull(src, B * (K + 1))
        x = torch.tensor([toks[i * (K + 1):i * (K + 1) + K] for i in range(B)], dtype=torch.long, device=dev)
        y = torch.tensor([toks[i * (K + 1) + 1:i * (K + 1) + K + 1] for i in range(B)], dtype=torch.long, device=dev)

        if args.mode == "curate":
            # ── LOOK BEFORE YOU LEARN (David's library: peek, then skip-or-learn) ──
            # Forward WITHOUT grad to score this block. The recurrent state still advances (free
            # under O(1) — that's the whole point), so context is never lost even on skipped blocks.
            # Train only if the block is "worth it": loss above an adaptive bar (a running quantile
            # of recent block losses) = the model doesn't get it yet = learn. Below = already known
            # / noise = SKIP (state advances, no gradient, no compute spent). Budget = TRAINED tokens.
            with torch.no_grad():
                peek_logits, peek_states = model(x, states)
                block_loss = float(lossf(peek_logits.reshape(-1, peek_logits.size(-1)), y.reshape(-1)))
            recent_losses.append(block_loss)
            if len(recent_losses) > args.curate_window:
                recent_losses.pop(0)
            bar = sorted(recent_losses)[int(len(recent_losses) * args.curate_skip_frac)] if len(recent_losses) > 8 else -1.0
            seen_tok += B * K
            if block_loss >= bar:                    # worth learning → real grad step
                logits, states = model(x, states)
                loss = lossf(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                states = [s.detach() for s in states]
                n_tok += B * K                       # TRAINED tokens (the fair budget)
                del logits, loss
            else:                                     # skip → advance state only, no learning
                states = [s.detach() for s in peek_states]
                skipped += 1
            del peek_logits, peek_states
        elif args.mode == "curate-replay":
            # NOTHING is discarded (the 3 negatives proved discarding always hurts). Warmup first
            # (learn to ride the bike, passive), THEN curate by REPLAY: learn every block once, and
            # ADDITIONALLY re-practice the hardest recent blocks (high loss = not mastered yet). The
            # "index" of what to revisit lives OUTSIDE the O(1) state — a small external replay deck,
            # consulted on demand, never carried in the bounded state. More practice for the hard
            # stuff, zero data thrown away.
            n_tok += B * K
            logits, states = model(x, states)
            loss = lossf(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            bl = float(loss)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            states = [s.detach() for s in states]
            del logits, loss
            if n_tok >= args.warmup_tokens:          # after warmup: build + use the replay deck
                replay_deck.append((bl, x.detach(), y.detach()))
                if len(replay_deck) > args.replay_size:
                    replay_deck.sort(key=lambda t: t[0])      # keep the HARDEST blocks
                    replay_deck.pop(0)
                if len(replay_deck) >= 8 and step % args.replay_every == 0:
                    _, rx, ry = max(replay_deck, key=lambda t: t[0])   # re-practice the hardest
                    rl, _ = model(rx, None)                    # stateless replay (a revisit, not the stream)
                    rloss = lossf(rl.reshape(-1, rl.size(-1)), ry.reshape(-1))
                    opt.zero_grad(set_to_none=True); rloss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                    replayed += 1
                    del rl, rloss
        else:
            n_tok += B * K
            logits, states = model(x, states)
            loss = lossf(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            states = [s.detach() for s in states]   # the O(1) carry
            del logits, loss
        peak = max(peak, _rss_gb())
        step += 1

        if step % args.eval_every == 0:
            hl, per = heldout_multi(model, heldout, K, dev)
            curve.append((n_tok, round(hl, 4), round(peak, 3),
                          {n: round(per[n], 3) for n in names}))
            extra = ""
            if args.mode == "curate":
                extra = f" | seen {seen_tok:,} trained {n_tok:,} skip-rate {skipped/max(1,step):.0%}"
            print(f"    trained {n_tok:>9,} | mean held-out {hl:6.3f} | rss {peak:4.2f}GB{extra}", flush=True)

    dt = time.time() - t0
    results = {"mode": args.mode, "sources": names, "d_model": args.d_model, "batch": B,
               "chunk": K, "tokens": n_tok, "elapsed_s": round(dt, 1),
               "base_heldout": round(base, 4), "final_heldout": round(curve[-1][1], 4),
               "pull_mix": {n: pulls[n] for n in names},
               "seen_tok": seen_tok, "skipped_blocks": skipped,
               "peak_rss_gb": round(peak, 3), "curve": curve}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n→ {args.out}")
    print(f"HEADLINE [{args.mode}]: mean held-out {base:.3f}→{curve[-1][1]:.3f} over {n_tok:,} tokens "
          f"at {peak:.2f}GB; pull-mix {results['pull_mix']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["passive-fixed", "passive-random", "active", "active-lp", "curate", "curate-replay"], default="curate-replay")
    ap.add_argument("--sources", default="c4,wiki,book")
    ap.add_argument("--lp-temp", type=float, default=0.05, help="(active-lp) softmax temp over learning-progress")
    ap.add_argument("--curate-skip-frac", type=float, default=0.5, help="(curate) skip blocks below this loss-quantile")
    ap.add_argument("--curate-window", type=int, default=200, help="(curate) rolling window for the skip bar")
    ap.add_argument("--warmup-tokens", type=int, default=500000, help="(curate-replay) passive warmup before curation")
    ap.add_argument("--replay-size", type=int, default=64, help="(curate-replay) bounded external replay deck size")
    ap.add_argument("--replay-every", type=int, default=4, help="(curate-replay) re-practice a hard block every N steps")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--block", type=int, default=32768)
    ap.add_argument("--target-tokens", type=int, default=2_000_000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--probe-every", type=int, default=20, help="(active) re-measure surprise every N steps")
    ap.add_argument("--mem-hard-gb", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/active_sourcing.json")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
