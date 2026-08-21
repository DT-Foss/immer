#!/usr/bin/env python3 -u
"""
Self-verify the billion-token streaming run from its result JSON.
=================================================================
Reads results/scale_to_a_billion.json and checks the claims the run is supposed to
demonstrate, printing PASS/FAIL for each and exiting 0 only if all pass. This is the
"exit 0 on success" companion to the other contributions — anyone can run it against the
committed JSON to confirm the headline without re-running the 1B stream.

Claims checked:
  1. The stream reached at least 1,000,000,000 tokens of effective sequence length.
  2. Peak memory stayed flat: max RSS over all checkpoints is within a small band of the
     first checkpoint (constant memory — the whole point).
  3. Perplexity stayed flat: the running PPL never drifts more than a small fraction from
     its first checkpoint across the entire billion-token span (no length wall).
  4. The PPL estimate rests on a huge sample (>>1e7 scored tokens).

Usage:  python src/verify_billion.py [path-to-json]
"""
import os, sys, json

DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "scale_to_a_billion.json")

# tolerances — generous enough to be honest, tight enough to be meaningful
TARGET_TOKENS = 1_000_000_000
RSS_BAND_GB   = 1.0      # peak RSS must stay within 1 GB of the first checkpoint
PPL_BAND_FRAC = 0.10     # running PPL must stay within 10% of the first checkpoint
MIN_SCORED    = 10_000_000


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.exists(path):
        print(f"FAIL: result file not found: {path}")
        print("      (run src/scale_to_a_billion.py first)")
        sys.exit(1)
    d = json.load(open(path))

    ckpts = d.get("checkpoints", [])              # [[streamed, ppl, rss], ...]
    streamed_final = d.get("tokens_streamed", 0)
    scored_final = d.get("tokens_scored", 0)
    if not ckpts:
        print("FAIL: no checkpoints in result")
        sys.exit(1)

    rss_vals = [c[2] for c in ckpts] + [d.get("peak_rss_gb", ckpts[-1][2])]
    ppl_vals = [c[1] for c in ckpts] + [d.get("final_ppl", ckpts[-1][1])]
    rss0, ppl0 = ckpts[0][2], ckpts[0][1]

    checks = []
    # 1. reached a billion
    ok = streamed_final >= TARGET_TOKENS
    checks.append(("reached >= 1B streamed tokens",
                   ok, f"{streamed_final:,} tokens"))
    # 2. memory flat
    rss_span = max(rss_vals) - min(rss_vals)
    ok = (max(rss_vals) - rss0) <= RSS_BAND_GB and rss_span <= RSS_BAND_GB
    checks.append(("peak memory flat (constant O(1) state)",
                   ok, f"RSS {min(rss_vals):.1f}-{max(rss_vals):.1f} GB "
                       f"(span {rss_span:.2f} GB, band {RSS_BAND_GB} GB)"))
    # 3. PPL flat
    max_drift = max(abs(p - ppl0) / ppl0 for p in ppl_vals)
    ok = max_drift <= PPL_BAND_FRAC
    checks.append(("running PPL flat across the whole span (no length wall)",
                   ok, f"PPL {min(ppl_vals):.1f}-{max(ppl_vals):.1f} "
                       f"(max drift {max_drift*100:.2f}% from {ppl0:.1f}, "
                       f"band {PPL_BAND_FRAC*100:.0f}%)"))
    # 4. huge sample
    ok = scored_final >= MIN_SCORED
    checks.append(("PPL estimate from a huge sample",
                   ok, f"{scored_final:,} scored tokens"))

    print(f"verifying {path}\n"
          f"  corpus={d.get('corpus','?')}  train_T={d.get('train_T','?')}  "
          f"chunk={d.get('chunk','?')} overlap={d.get('overlap','?')} "
          f"batch={d.get('batch','?')}  checkpoints={len(ckpts)}\n")
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")

    extrap = streamed_final // max(1, d.get("train_T", 32))
    print(f"\n  HEADLINE: {streamed_final:,} tokens ({extrap:,}x training length) "
          f"at constant {max(rss_vals):.1f} GB, PPL flat at ~{ppl0:.0f}.")
    print(f"\n{'ALL CHECKS PASS' if all_ok else 'SOME CHECKS FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
