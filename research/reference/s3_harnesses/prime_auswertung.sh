#!/bin/bash
# A5-prime Auswertung: fetch + merge + redistill + verifier_loop (Mac)
set -u
cd /Users/bhkmie/self-verification_fable/mirkoNN-analysis/s3
mkdir -p /tmp/27b/prime
scp -o ConnectTimeout=12 -o BatchMode=yes 'root@100.119.16.99:/root/donor_chunk_prime_*.npz' /tmp/27b/prime/
python3 merge_27b_chunks.py /tmp/27b/prime/donor_chunk_prime_*.npz \
  -o /tmp/27b/prime/donor_targets_prime_27b.npz
# Donor-Qualitaet: yes/no-Verteilung
python3 - <<'PYEOF'
import numpy as np
d = np.load("/tmp/27b/prime/donor_targets_prime_27b.npz", allow_pickle=True)
for key, a in (("logits", d["ans"]), ("logits_eval", d["ans_eval"])):
    lg = d[key][:, 0, :]
    e = np.exp(lg - lg.max(1, keepdims=True)); e /= e.sum(1, keepdims=True)
    ok = (e.argmax(1) == np.array([1 if x == "no" else 0 for x in a])).mean()
    print(f"donor prime {key}: acc {ok:.3f} | cand={list(d['cand'])}")
PYEOF
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_NO_ADVISORY_WARNINGS=1
python3 redistill.py --donor-npz /tmp/27b/prime/donor_targets_prime_27b.npz \
  --n-pairs 256 --n-eval 96 --steps 600 --loss digkl --cand "yes,no" \
  --task prime --out redistill_prime.json 2>&1 | grep -v "Loading\|Materializing\|Warning\|W08" | tail -9
