#!/usr/bin/env bash
set -euo pipefail

: "${DONOR_MODEL:?set DONOR_MODEL to the local GGUF checkpoint}"
: "${DONOR_PYTHON:?set DONOR_PYTHON to the llama.cpp Python interpreter}"
: "${DONOR_HARNESS:?set DONOR_HARNESS to donor_chunk.py}"
: "${DONOR_PROMPTS:?set DONOR_PROMPTS to the prompt JSON}"
: "${DONOR_OUTPUT_DIR:?set DONOR_OUTPUT_DIR to a writable run directory}"
: "${DONOR_TASK:?set DONOR_TASK to the output stem}"
: "${DONOR_OUTPUT_PREFIX:?set DONOR_OUTPUT_PREFIX to the historical file prefix}"

mkdir -p "$DONOR_OUTPUT_DIR"
LOG_PATH="$DONOR_OUTPUT_DIR/donor_chunks_${DONOR_TASK}.log"
: >"$LOG_PATH"

runs=(
  "train 0 64"
  "train 64 64"
  "train 128 64"
  "train 192 64"
  "eval 0 64"
  "eval 64 32"
)

for spec in "${runs[@]}"; do
  read -r split offset count <<<"$spec"
  output="$DONOR_OUTPUT_DIR/${DONOR_OUTPUT_PREFIX}_${split}_${offset}.npz"
  printf '== %s off=%s cnt=%s ==\n' "$split" "$offset" "$count" >>"$LOG_PATH"
  command=(
    "$DONOR_PYTHON" -u "$DONOR_HARNESS"
    --model "$DONOR_MODEL"
    --prompts "$DONOR_PROMPTS"
    --set "$split"
    --offset "$offset"
    --count "$count"
    --out "$output"
  )
  if [[ -n "${DONOR_CANDIDATES:-}" ]]; then
    command+=(--cand "$DONOR_CANDIDATES")
  fi
  "${command[@]}" >>"$LOG_PATH" 2>&1
done

printf 'ALL DONE\n' >>"$LOG_PATH"
