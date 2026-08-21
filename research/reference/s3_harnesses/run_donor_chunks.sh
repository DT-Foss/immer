#!/bin/bash
set -u
MODEL=/root/o1x_data/qwen38-27b-gguf/Qwen3.8-27B-Q3_K_M.gguf
PY=/root/venv-llama/bin/python
cd /root
rm -f /root/donor_chunks_run.log /root/donor_chunk_*.npz
for spec in "train 0 64" "train 64 64" "train 128 64" "train 192 64" "eval 0 64" "eval 64 32"; do
  set -- $spec
  SET=$1; OFF=$2; CNT=$3
  echo "== $SET off=$OFF cnt=$CNT ==" >> /root/donor_chunks_run.log
  $PY -u /root/donor_chunk.py --model $MODEL --prompts /root/27b_add2_prompts.json --set $SET --offset $OFF --count $CNT --out /root/donor_chunk_${SET}_${OFF}.npz >> /root/donor_chunks_run.log 2>&1
  echo "exit=$?" >> /root/donor_chunks_run.log
done
echo "ALL DONE" >> /root/donor_chunks_run.log
