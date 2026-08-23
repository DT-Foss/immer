# Roadmap

Release line: 0.8.x · 2026-08-23

The target is a local frontier runtime whose immutable weights are fully
addressable through a growing causal control plane, with exact verification
around every domain that supports it.

## Complete

- [x] Full DeepSeek-V4-Flash checkpoint preflight and 43-layer decoder.
- [x] Layer-major multi-item execution with authenticated resume.
- [x] Exact local and pinned-remote Safetensors range readers.
- [x] Bounded disk cache, integrity checks, byte receipts, and asynchronous
  expert transport.
- [x] LiveCausal append-only graph with lazy query, citations, tombstones,
  crash recovery, and concurrent-reader refresh.
- [x] Causalized local bundle mount: immutable `weights/` plus live `causal/`.
- [x] Official expert routes bound to exact range plans and consumed by the
  DeepSeek pager.
- [x] Label-free token-row Markov predictor with full score distributions,
  configurable \(k\), held-out prompt splits, and placebo evaluation.
- [x] FERTIG exact cascade, structural IR, guarded fallbacks, and zero-wrong
  GSM8K audit.
- [x] Own strict-causal Local/Balanced/Free attention and frozen context router.
- [x] Frozen SHIP-v6 host with four digest-addressed organs.
- [x] O(1)-state life stream separated from frozen execution.

## Active

- [ ] Build a full local DeepSeek causal bundle from the pinned checkpoint and
  populate all expert rails.
- [ ] Run the first end-to-end local A/B/C: plain paging, learned causal
  routing, and shuffled causal placebo on identical prompts.
- [ ] Report Recall@\(k\), precision@\(k\), bytes requested, bytes resident,
  cache churn, source wait, wall time, and output equality.
- [ ] Adapt \(k\) and prefetch depth to observed bandwidth, memory pressure,
  and transition entropy instead of fixing one global width.
- [ ] Feed completed route observations back into LiveCausal during ordinary
  inference and verify immediate reader visibility.
- [ ] Extend the benchmark from integration slices to representative MMLU and
  GSM8K cohorts with fixed dataset hashes.
- [ ] Make DeepSeek the default neural component in the composition root after
  the local causal A/B/C passes.

## Release gates

- [ ] Reproduce every public result from a clean checkout.
- [ ] Run the complete test suite with ResourceWarnings promoted to errors.
- [ ] Verify every documentation link and every referenced repository path.
- [ ] Build wheel and source distribution for version 0.8.0.
- [ ] Scan the release tree for weights, caches, secrets, private traces,
  machine topology, and experimental transfer material.
- [ ] Produce the offline model bundle and its checksum allowlist.
- [ ] Publish only after model and component licenses are explicitly cleared.
