# Changelog

All notable changes to IMMER are recorded here.

## [0.8.0] — 2026-08-23

### Causalized local frontier runtime

- Completed the DeepSeek-V4-Flash 43-layer decoder with layer-major scoring,
  authenticated resume, stateful generation, exact range paging, and a
  guarded chat adapter.
- Added local positional Safetensors reads so production inference can run
  from local checkpoint storage without a second payload cache.
- Added LiveCausal: content-addressed append-only segments, a hash-chained
  journal, atomic head commits, lazy exact queries, citations, reversible
  tombstones, crash recovery, and concurrent-reader refresh.
- Added the causalized model bundle: immutable `weights/` and append-only
  `causal/` mounted under one checkpoint-bound identity.
- Bound official DeepSeek expert coordinates to exact local range plans and
  connected causal range resolution to the existing expert pager.
- Added exact access tracing and deterministic range replay with checkpoint
  identity verification and configurable replay windows.
- Added label-free token-row Markov routing, full expert distributions,
  configurable top-k evaluation, prompt-held-out splits, target-marginal
  baselines, and label-preserving placebos.
- Executed Markov hints through byte-budgeted expert reservoirs with measured
  confidence gating, direct decode windows, deduplicated large-prefill vote
  aggregation, exact miss fallback, and bit-identical three-arm comparison.
- Added transport-neutral, model-math-bound KV snapshots and a contextual
  shared-prefix decode benchmark for strict baseline/real/placebo TPOT arms.
- Built trace-complete sparse causal bundles with byte-identical Safetensors
  coordinates, exact reconstructed headers, pinned inventory identity, and
  fail-closed bindings for every fully materialized expert.
- Executed shared-prefix decode directly through the local causal reader with
  bit-identical hidden states: about 53 seconds locally versus 121 seconds from
  a warm remote range cache and 1,100+ seconds from cold remote transport.
- Added FERTIG draft verification and exact/model fusion receipts for fixed
  integration cohorts.
- Completed Qwen3.8 stateful execution: all full-attention KV and DeltaNet
  recurrent/convolution states now persist across prefill and decode, failed
  forwards poison and clear partial state, CRSA history continues causally,
  and streamed greedy generation leaves a fully resumable cursor.
- Proved batched prefill, tokenwise prefill, and continuation decode are
  bit-identical in the checkpoint's BF16 execution mode on the executable
  model fixture.
- Added authenticated Qwen continuation snapshots for every KV, DeltaNet,
  cursor, batch, and CRSA-history state. The pickle-free manifest binds its
  schema, checkpoint layout, config, math runtime, dependencies, and graft;
  the content-addressed payload is verified before bounded transactional
  restore.
- Generalized the causal weight graph from DeepSeek expert coordinates to
  immutable tensor coordinates with live-append bindings, revision-bound plan
  caches, exact subrange reads, conflict detection, and tombstone invalidation.
- Connected `Qwen38WeightPager` to the causal tensor reader. Dense prefill,
  row-paged LM-head scans, and stateful greedy generation now execute without
  tensor metadata discovery and remain bit-identical to the inventory path.
- Added the complete Qwen causal-bundle builder: every shard is size- and
  SHA-256-verified against the pinned inventory before tensor bindings are
  published; resumable staging, config/index validation, full graph replay,
  post-copy verification, and atomic promotion prevent sparse holes or partial
  checkpoints from becoming executable model bytes.
- Added in-place bundle adoption for storage-constrained deployments: an
  existing complete `weights/` tree is fully re-hashed and causalized beside
  the unchanged shards, eliminating the second 55.6-GB checkpoint copy.
- Added pinned `fetch-adopt` provisioning: official shards download directly
  into their final `weights/` tree with exact range resume, crash-safe partial
  recovery, full SHA-256 verification, and zero-copy causal adoption.
- Made Qwen layer-major verification emit complete resumable access traces.
  Content-addressed trace and hidden-resume files commit through one hashed
  pair manifest, resumed traces renumber new operations canonically,
  identity/capacity drops fail the run, and the final result cites its exact
  trace digest and coverage.
- Added the Qwen shared-prefix direct-decode benchmark: input selection,
  stateful prefix publication, transport-neutral continuation restore, exact
  per-arm access traces, hidden dtype/shape/hash invariants, inventory-versus-
  causal execution, and strict remote/local timing comparison.
- Fixed Qwen continuation admission on Apple MPS: indexless pager device
  `mps` now correctly accepts resolved tensor device `mps:0`, with an actual
  MPS BF16 save/restore/decode regression test.
- Ported the measured Qwen3.8-27B DeltaNet instrument into the exact runtime:
  nine passive per-layer signals, authenticated prefix/decode probe artifacts,
  and pooled-sample Cohen-d maps now calibrate graft placement from contextual
  model states without changing inference output.
- Rebuilt public documentation around the local causal architecture and
  separated public source from weights, state, traces, and operations.

## [0.7.0] — 2026-08-22

### Frozen exact core and reproducible transport

- Integrated the frozen SHIP-v6 host and four SHA-addressed exact organs.
- Established `ExactCascade` as the single exact-arithmetic owner with FERTIG
  verification, grounded fallback, and abstention.
- Integrated role-complete Causal Prefix Sinkhorn Attention (CRSA) and the
  persisted context router.
- Added atomic artifact import, digest verification, bounded Safetensors range
  access, and the offline export builder.
- Added full GSM8K FERTIG audit, structural IR certificates, and public
  machine-readable benchmark receipts.
- Implemented and measured exact routed-expert prefetch, adjacent range
  envelopes, and LM-head range batching.

## [0.6.0] — 2026-08-21

### Living local composition

- Added the persistent O(1)-state life stream, surprise-gated updates, sleep
  consolidation, replay state, and service dashboard.
- Vendored FERTIG behind a runtime adapter and explicit action gates.
- Added the capability bank, growing local knowledge library, council, local
  neural mouth, and command-line service loop.
- Added the first Qwen3.8 range-streaming, router, and FERTIG fusion
  experiments.

## [0.1.0] — 2026-08-20

- Created the IMMER integration shell, contracts, registry, substrate, and
  initial composition root.
