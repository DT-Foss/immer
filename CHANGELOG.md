# Changelog

All notable changes to IMMER are recorded here.

## [Unreleased] — 2026-08-24

### Transactional Qwen continuation blocks

- Added opaque stage/commit/discard transactions for multi-token continuation
  blocks without exposing commit-state tensors to the verifier.
- Delayed state, graft history, and observers until atomic commit; stale,
  foreign, mutated, reset, failed, and runtime-drifted stages fail closed.
- The first official matrix-block replay halved bytes and runtime but
  falsified state parity at layer 0. Replaced it with a layer-major K=2 path
  that reads every weight once while executing the same two one-token kernels;
  F.linear calls remain 62 and tensor reads halve.
- Proved the exact K=2 path bit-identical on a 64-layer/24Q/4KV CPU-BF16
  fixture for off, stable graft, and native Prefix-Sinkhorn states.
- Added tokenwise-exact Prefix-Sinkhorn usage updates inside staged blocks;
  tiny CPU BF16 native K=2 now matches tokenwise hidden and every state bit.
- Hard-bounded the public transaction to the proven `batch=1, K=2` tranche;
  larger blocks and batches fail closed until their DeltaNet arithmetic uses a
  weight-once/tokenwise-linear path. Ordinary decoding is unchanged.
- Added a gold-free official parity harness that replays a sealed off-arm
  `[token, EOS]` chain through two fresh causal-bundle runtimes, hashes every
  hidden/KV/DeltaNet/graft state, authenticates both access traces, and seals
  either a positive or mismatch result.
- The official 64-layer CPU-BF16 replay is positive: every hidden and
  continuation-state hash matches, continuation source bytes fall by
  `49.99999%` (`97.41 GB → 48.71 GB`), and model time falls by `46.48%`
  (`158.14 s → 84.63 s`) while preserving all 992 one-token linear calls.
- Added exact K=2 speculative generation over an arbitrary draft provider:
  full acceptance commits once, mismatch-0 decodes the target token,
  mismatch-1 restages the verified prefix, and EOS never commits post-stop
  state. Every path is bit-identical to greedy generation in regression tests.
- Hardened same-process draft hooks with before/after committed-state hashing;
  mutation resets and fails closed. Integrity cost is explicit in
  `provider_guard_bytes` and `provider_guard_seconds`, and speculative
  generation exposes no live progress callbacks around pending state.

### Exact event and binding compiler

- Added a span-aware quantitative event frontend that lowers twenty-seven typed
  transaction, rate, comparison, residual, and repeated-duration families to
  exact signed-expression DAGs.
- Added exact calendar-rate binding for fixed 30/31-day months; February and
  multi-month questions remain fail-closed without an explicit day basis.
- Added unique object-possessive binding for original-length relations without
  global pronoun guessing.
- Raised the question-only 64-item exact-certificate frontier from 3 to 31;
  every other item remains an abstention.
- Upgraded the gold-free abstention audit to partition exact recoveries from
  remaining ambiguity and unsupported grammar. The current report proves 30
  recoveries from the previous 230 abstentions, leaves 200 fail-closed, and
  binds the event/DAG compiler hashes in report provenance.
- Added adversarial coverage for cross-owner, cross-item, foreign-price,
  duplicate-share, numeric-noise, reordered-clause, and possessive-scope
  attacks.

### Certificate-first Qwen/FERTIG fusion

- Added 16 full-string guarded formula families with exact `Fraction`
  recomputation, complete numeric-literal coverage, source spans, and
  SHA-256-bound certificates.
- Moved guarded formulas and structural Fraction/RREF ahead of the legacy
  FERTIG solver, including an explicit conflict gate between independent exact
  certificates.
- Raised full GSM8K coverage from 81.35% to 82.56%: 1,089 correct, zero wrong,
  and 230 abstentions across all 1,319 test rows.
- Rejected Q3/BF16 checkpoint agreement as an answer certificate after a
  16-item hard cohort exposed six shared wrong answers. Model agreement now
  quarantines by default; the legacy policy is available only as an explicit
  diagnostic switch.
- Completed the hard cohort with 16/16 exact answers, zero wrong, and 100%
  coverage. Gold labels enter only after each answer or abstention is fixed.
- Enforced dynamic-cohort status semantics, sealed-report integrity, and the
  shared 64-item producer/consumer limit.
- Preserved bounded truncated drafts through teacher-forced verification as
  explicitly incomplete, non-answer candidates instead of dropping their
  preselected cohort rows.
- Added sealed dynamic-cohort offsets so development and holdout slices remain
  explicitly disjoint while earlier FERTIG abstentions become certified.
- Added a fail-closed paired `off`/CRSA comparator that requires identical
  checkpoint, causal bundle, cohort, prompts, and drafts, and treats new
  agreement with wrong drafts as an unsafe regression rather than quality.
- Added authenticated common-prefix forking at the exact CRSA graft boundary.
  The candidate inherits the off arm's hidden state, complete causal trace,
  source bytes, and model time, then recomputes every changed layer.
- Added an evidence-closed local clause compiler foundation with typed entity,
  item, scope, state, numeric-span, relation, and target records. Its first
  general family lowers affine count systems to the existing exact IR without
  overriding prior ambiguity or invalidity decisions.

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
- Added flat in-place adoption for checkpoint directories that already hold
  the shards: the authenticated inventory, manifest, and append-only causal
  rail are implanted beside the original files without copying, moving,
  hardlinking, or changing any weight inode or byte.
- Added pinned `fetch-adopt` provisioning: official shards download directly
  into their final `weights/` tree with exact range resume, crash-safe partial
  recovery, full SHA-256 verification, and zero-copy causal adoption.
- Corrected Hugging Face Xet identity handling: the CDN/Xet file ID remains a
  transport pin while `X-Linked-ETag` supplies the reconstructed payload
  SHA-256. Inventory refresh, local adoption, and downloads now bind both and
  never mistake an Xet ID for a shard-byte digest.
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
- Promoted complete Qwen bundle verification into the runtime library and
  connected the FERTIG draft benchmark to nested or flat causal bundles. Local
  quality arms now re-hash every payload shard, replay every tensor binding,
  and execute through the causal tensor reader instead of HF transport.
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
