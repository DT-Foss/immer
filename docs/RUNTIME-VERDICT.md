# Runtime Verdict

Release: 0.8.0 · 2026-08-23

## Verdict

IMMER is a working local systems prototype with four proven execution planes:

1. a complete DeepSeek-V4-Flash decoder that reads exact Safetensors ranges;
2. a causalized local bundle that joins immutable weights to append-only
   LiveCausal routes;
3. a deterministic FERTIG/SHIP path that answers or abstains under explicit
   verification;
4. a persistent O(1)-state life stream isolated from frozen execution.

The active engineering target is local causal weight transport. The graph and
reader path are integrated and tested. The end-to-end local speed comparison
is the remaining measurement gate.

## Proven in committed evidence

- DeepSeek completes all 43 layers on a fixed four-item MMLU slice and scores
  3/4 in exact `off` mode.
- FERTIG scores 1,073 correct, zero wrong, and 246 abstentions across all 1,319
  GSM8K test rows.
- SHIP-v6 scores 152/152 answers and 152/152 routes with four frozen organs.
- The CRSA context router scores 182/182 on its fixed arithmetic/text corpus;
  the causal-softmax ablation matches it.
- The static Qwen value-sketch shortcut scores below placebo and is absent from
  runtime routing.
- Remote transport experiments prove bit-identical range batching and expose
  the latency behavior of each transport policy.

See [`BENCHMARKS.md`](BENCHMARKS.md) for every receipt and scope sentence.

## Implemented in the runtime

| Plane | Implementation | Runtime contract |
|---|---|---|
| DeepSeek V4 | `src/immer/runtimes/deepseek_v4/` | exact checkpoint math, layer-major execution, resume, generation |
| LiveCausal | `src/immer/knowledge/livecausal.py` | append-only durable graph, lazy exact query, citations |
| causal weights | `src/immer/runtimes/deepseek_v4/causal_weights.py` | checkpoint-bound semantic coordinate to exact local ranges |
| Markov routing | `src/immer/runtimes/deepseek_v4/route_markov.py` | label-free next-layer expert distribution and placebo evaluation |
| local range I/O | `src/immer/knowledge/streamer.py` | positional reads, integrity, accounting, optional cache |
| exact cascade | `src/immer/cognition/exact_cascade.py` | S3 first, FERTIG verification/fallback, abstention |
| Causal Prefix Sinkhorn Attention | `src/immer/attention/` | strict-causal Local/Balanced/Free roles |
| life stream | `src/immer/runtimes/o1_state/` | persistent adaptive state separate from frozen paths |

## Scope boundary

Version 0.8.0 establishes the architecture and its component-level proofs. A
frontier-equivalent quality claim requires broad fixed benchmarks. A local
causal speed claim requires the pending plain/real/placebo end-to-end report.
The release states neither claim before those receipts exist.

## Public release boundary

Public source includes algorithms, runtime contracts, tests, manifests, and
small benchmark receipts. Model weights, caches, live causal graphs, private
route traces, operational topology, credentials, and experimental transfer
artifacts stay outside Git.
