# Benchmarks

Release: Unreleased · 2026-08-25

Each row states its exact measurement scope and evidence boundary. Committed
machine-readable receipts are linked directly. Model-weight execution receipts
that remain private are identified as private and paired with the committed
harness and validation tests that define their schema. Transport
microbenchmarks are separated from model-quality benchmarks, and
remote-network results are separated from local execution.

## Runtime and quality

| Benchmark | Result | Scope | Evidence |
|---|---:|---|---|
| SHIP-v6 exact suite | 152/152 answers; 152/152 routes | frozen host plus four exact organs | `manifests/s3_ship_v6.json`; `python -m immer eval` |
| FERTIG GSM8K | 1,089 correct; 0 wrong; 230 abstentions; 82.56% coverage | all 1,319 GSM8K test rows, certificate-first and strict zero-wrong scoring | [`results/bench_gsm8k.json`](../results/bench_gsm8k.json) |
| FERTIG exact Dev frontier | 45/64 certified; 19 abstained | fixed gold-free question-only development slice; holdout remains untouched; selection receipt private | [`signed_event_frontend.py`](../src/immer/cognition/fertig/signed_event_frontend.py); [`test_fertig_signed_event_frontend.py`](../tests/test_fertig_signed_event_frontend.py) |
| FERTIG abstention recovery | 47/230 recovered; 183 remain abstained | gold-free reclassification of the previously sealed 230 abstentions; 42 prior recoveries plus 5 from the current wave | [`results/fertig-abstention-audit.json`](../results/fertig-abstention-audit.json); [`fertig_abstention_audit.py`](../scripts/fertig_abstention_audit.py) |
| DeepSeek V4 MMLU | 3/4 correct; 43/43 layers complete | exact `off` mode on one fixed four-item high-school-geography slice | [`results/deepseek-v4-mmlu-off-4-exact-v4-window2x3-receipt.json`](../results/deepseek-v4-mmlu-off-4-exact-v4-window2x3-receipt.json) |
| DeepSeek V4 + FERTIG | 6/8 correct; 0 wrong; 2 abstentions | fixed eight-item GSM8K failure slice; five exact certificates plus one model-verified answer | [`results/deepseek-v4-fertig-fusion.json`](../results/deepseek-v4-fertig-fusion.json) |
| Qwen3.8 + FERTIG | 5/8 exact; 0 wrong; 3 quarantined | fixed eight-item GSM8K failure slice; model agreement cannot answer without an independent certificate | [`results/qwen38_fertig_fusion.json`](../results/qwen38_fertig_fusion.json) |

The four-item and eight-item rows are integration slices. Their claims stop at
those exact item sets.

## Local causal Qwen continuation

| Benchmark | Result | Exact scope | Evidence boundary |
|---|---:|---|---|
| Qwen3.8 exact K=2 continuation | state and hidden bit-identical; `49.999989%` fewer source bytes; `46.483087%` less model time; `1.868568x` speedup | one fixed `[token, EOS]` continuation, two fresh 64-layer CPU-BF16 causal-bundle runtimes; all KV and DeltaNet state hashes equal | private sealed receipt; [`qwen38_continuation_block_parity.py`](../scripts/qwen38_continuation_block_parity.py); [`test_qwen38_continuation_block_parity.py`](../tests/test_qwen38_continuation_block_parity.py) |
| Qwen3.8 fixed-Q3 speculative K=2 | 2/2 accepted; `33.893734%` fewer target source bytes; `22.435086%` less model time; `1.289243x` speedup | one fixed gold-free Q3 pair; greedy and speculative tokens, hidden, cursor, KV, and DeltaNet state equal | private sealed receipt; [`qwen38_speculative_k2_benchmark.py`](../scripts/qwen38_speculative_k2_benchmark.py); [`test_qwen38_speculative_k2_benchmark.py`](../tests/test_qwen38_speculative_k2_benchmark.py) |
| Live Qwen3.5-0.8B → Qwen3.8-27B K=2 | 2/2 accepted; `31.244579%` fewer combined target-plus-drafter source bytes; `23.029270%` less wall time; `1.299195x` speedup | one fixed two-token, same-prompt CPU-BF16 native Prefix-Sinkhorn trial; target output `[760, 6511]`; target alone commits | private sealed live receipt plus an immediate same-runtime greedy control; [`qwen35_live_k2_smoke.py`](../scripts/qwen35_live_k2_smoke.py); [`test_qwen35_live_k2_smoke.py`](../tests/test_qwen35_live_k2_smoke.py); [`test_qwen35_local_draft.py`](../tests/test_qwen35_local_draft.py) |

The live trial compares `151,204,957,184` greedy target bytes with
`99,955,757,056` live target bytes plus `4,005,847,104` drafter bytes. Wall
time changes from `243.376159193 s` to `187.328407226 s`. This is a positive
mechanism and fixed-prompt speed result, not a cohort acceptance or quality
claim.

Run the committed contract tests:

```bash
PYTHONPATH=src:tests python -m unittest \
  test_qwen38_continuation_block_parity \
  test_qwen38_speculative_k2_benchmark \
  test_qwen35_local_draft \
  test_qwen35_live_k2_smoke
```

The live harness accepts explicit causal-bundle paths and writes one sealed
result:

```bash
PYTHONPATH=src python scripts/qwen35_live_k2_smoke.py \
  --target-bundle <qwen3.8-causal-bundle> \
  --draft-bundle <qwen3.5-causal-bundle> \
  --prompt 'The capital of France is' \
  --max-new-tokens 2 \
  --attention-mode native-crsa \
  --device cpu \
  --compute-dtype bfloat16 \
  --output <result.json>
```

## Causal Prefix Sinkhorn Attention and routing

| Measurement | Result | Scope |
|---|---:|---|
| CRSA context router | 182/182 arithmetic/text decisions | fixed router evaluation corpus |
| Raw frozen-host ablation | balanced accuracy 0.983333 | same corpus |
| Persisted context head | balanced accuracy 1.000000 | same corpus |
| 32 label placebos | mean 0.510328; max 0.748026 | same features and evaluation protocol |
| causal-softmax ablation | balanced accuracy 1.000000 | same router role, different attention normalizer |

The context signal is real. This corpus gives CRSA and causal softmax the same
balanced accuracy. The supported result is the router and its role program;
the two normalizers remain tied on this corpus.

Reproduce the report:

```bash
PYTHONPATH=src python scripts/crsa_route_eval.py
```

## Router-v2 structural scan

The placebo-controlled gate scan locates question structure in middle and
late-middle layers of Qwen3.8-27B. Layer 0 is dead; layers 18, 27, 36, 45,
and 54 exceed their placebo controls. The follow-on static value sketch scores
24% against a 32% placebo at 138.4 KB per question and is rejected.

Evidence:

- [`results/router_v2_stage1.json`](../results/router_v2_stage1.json)
- [`results/router_v2_stage2.json`](../results/router_v2_stage2.json)

## DeepSeek transport: remote network

These tests use the pinned official checkpoint over network range requests.
They measure transport envelopes only. All compared outputs are bit-identical.

| Experiment | Result | Scope |
|---|---:|---|
| one-ahead exact prefetch | 1.0928× mean speedup | ten routed-expert transport trials; cache-resident source | [`results/deepseek-v4-exact-prefetch-smoke.json`](../results/deepseek-v4-exact-prefetch-smoke.json) |
| window prefetch, cold network | 0.7330× mean speedup | four trials; exact expert ranges; candidate wins 3/4 but one network tail dominates the mean | [`results/deepseek-v4-exact-prefetch-window-network-smoke.json`](../results/deepseek-v4-exact-prefetch-window-network-smoke.json) |
| window prefetch, warm cache | 1.8311× mean speedup | ten trials; candidate wins 10/10 | [`results/deepseek-v4-exact-prefetch-window-smoke.json`](../results/deepseek-v4-exact-prefetch-window-smoke.json) |
| adjacent pair envelopes, cold network | 1.0763× mean speedup | ten trials; same bytes, fewer envelopes; candidate wins 7/10 | [`results/deepseek-v4-adjacent-range-network-smoke.json`](../results/deepseek-v4-adjacent-range-network-smoke.json) |
| adjacent pair envelopes, warm cache | 0.9822× mean speedup | ten trials; candidate wins 5/10 | [`results/deepseek-v4-adjacent-range-warm-smoke.json`](../results/deepseek-v4-adjacent-range-warm-smoke.json) |
| LM-head range batching | 1.4458× paired median; 4/4 wins | exact official LM head; 127 logical blocks; same 1,059,061,760 bytes | [`results/deepseek-v4-head-range-network-smoke.json`](../results/deepseek-v4-head-range-network-smoke.json) |

Network variance is visible in every report. The causalized local bundle
removes remote checkpoint transport from the production architecture.

## Local causal execution status

The local causal path now executes end to end. The repository contains and
tests:

- local positional Safetensors reads;
- append-only LiveCausal segments and crash recovery;
- checkpoint-bound causal weight plans;
- concurrent graph append with reader refresh;
- exact expert payload reads through graph routes;
- pager attachment for causal range resolution;
- complete stateful Qwen3.8 prefill, decode, and exact K=2 continuation;
- a causalized Qwen3.5-0.8B transactional draft provider;
- native Prefix-Sinkhorn target verification with target-only commits;
- token-row Markov prediction, held-out prompt splitting, \(k\)-sweeps, and
  label-preserving placebo construction.

The three local Qwen rows above are the current exact speed boundary. Their
receipts bind one continuation pair or one fixed prompt; broader throughput,
acceptance, and quality claims begin only after a frozen multi-prompt cohort.

## Reproduction gate

```bash
PYTHONPATH=src python -W error::ResourceWarning -m unittest discover -s tests
PYTHONPATH=src python -m compileall -q src scripts
git diff --check
```
