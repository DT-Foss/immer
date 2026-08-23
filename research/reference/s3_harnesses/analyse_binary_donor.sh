#!/usr/bin/env bash
set -euo pipefail

: "${S3_HARNESS_DIR:?set S3_HARNESS_DIR to this harness directory}"
: "${S3_WORK_DIR:?set S3_WORK_DIR to a writable temporary directory}"
: "${DONOR_SCP_SOURCE:?set DONOR_SCP_SOURCE to user@host:/path/glob}"
: "${REDISTILL_OUTPUT:?set REDISTILL_OUTPUT to the result JSON path}"
: "${DONOR_TASK:?set DONOR_TASK to the task name}"

S3_PYTHON="${S3_PYTHON:-python3}"
mkdir -p "$S3_WORK_DIR"
scp -o ConnectTimeout=12 -o BatchMode=yes "$DONOR_SCP_SOURCE" "$S3_WORK_DIR/"

shopt -s nullglob
chunks=("$S3_WORK_DIR"/donor_chunk_"$DONOR_TASK"_*.npz)
if (( ${#chunks[@]} == 0 )); then
  printf 'no donor chunks found for %s in %s\n' "$DONOR_TASK" "$S3_WORK_DIR" >&2
  exit 2
fi
merged="$S3_WORK_DIR/donor_targets_${DONOR_TASK}.npz"
"$S3_PYTHON" "$S3_HARNESS_DIR/merge_27b_chunks.py" "${chunks[@]}" -o "$merged"

"$S3_PYTHON" - "$merged" "$DONOR_TASK" <<'PYEOF'
import sys

import numpy as np

path, task = sys.argv[1:]
data = np.load(path, allow_pickle=True)
for key, answers in (("logits", data["ans"]), ("logits_eval", data["ans_eval"])):
    logits = data[key][:, 0, :]
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    gold = np.array([1 if value == "no" else 0 for value in answers])
    accuracy = (probs.argmax(1) == gold).mean()
    print(f"donor {task} {key}: acc {accuracy:.3f} | cand={list(data['cand'])}")
PYEOF

export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
"$S3_PYTHON" "$S3_HARNESS_DIR/redistill.py" \
  --donor-npz "$merged" \
  --n-pairs 256 \
  --n-eval 96 \
  --steps 600 \
  --loss digkl \
  --cand "yes,no" \
  --task "$DONOR_TASK" \
  --out "$REDISTILL_OUTPUT"
