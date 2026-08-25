# Changelog

All notable changes to IMMER are recorded here.

## [Unreleased] — 2026-08-25

### Qwen compute batteries

- Added exact native continuation batteries over authenticated Qwen snapshots.
  Active Prefix-Sinkhorn usage state, KV state and DeltaNet state now survive
  save/restore and remain bound to the complete runtime identity.
- Added a semantic anchor cache with deepest exact-prefix matching, atomic
  snapshot-first/index-commit recovery, deterministic LRU, byte budgets and
  explicit orphan collection. Exact-prefix cells carry authenticated final
  hidden state for direct LM-head scanning without replaying the last token.
- Added the demand-driven control plane: incremental radix prefix mining,
  semantic boundaries, O1 learning-progress signals, the complete charge/store/
  verify/invalidation profitability inequality, value-density selection, SoC,
  self-discharge and cache-turnover accounting.
- Added a sealed AB/BA/ABBA/BAAB harness that separates idle charging from peak
  demand and requires bit-identical final hidden and serialized native state.
- Added a FERTIG verifier that binds one honest replica vote to the complete
  cartography evidence and three bound proof surfaces without presenting
  those surfaces as a fake three-node quorum.
- Verified the complete integration with `1175/1175` tests.

### O1 semantic cartography over causal Qwen

- Reconnected the system's O1-State foundation to causal Qwen execution:
  persistent scheduling, prompt-level learning progress, replay, and resume now
  drive exact layer/module probes into an append-only SemanticWeightAtlas.
- Restored the canonical POS objective (`x_t -> x_(t+1)`) with continuous byte
  carry, surprise-gated plasticity, and content-addressed persistence of model,
  optimizer, recurrent Z-state, and stream tail.
- Added a multi-prompt `prepare`/`run`/`status`/`query` loop with immutable model
  pins, separate weight-rail and atlas revisions, exact tensor-range receipts,
  atomic appends, crash recovery, and complete prompt-by-coordinate frontiers.
- Added external semantic-label bindings for exact FERTIG proofs. Label source,
  semantic label, and proof digest are part of probe identity; model output has
  no label authority, and placebo controls never inherit the primary label.
- Added exact passive, off, native Prefix-Sinkhorn, and paired placebo probe
  modes. The first full-checkpoint execution completed both passive coordinates
  and the first paired native layer-27 coordinate; the intervention is identical
  before its hook and produces a measured nonzero post-layer change.
- Closed the native coordinate contract by binding the intervention to the
  runtime's atomic attention-head group; single-head coordinates are rejected.
- Completed the first live multi-prompt frontier: two prompts by two coordinates,
  four successful jobs in one Qwen process. O1 learning progress selected the
  remaining sibling coordinate of the same prompt next, and the FERTIG-backed
  semantic label reopened with both its passive observation and causal native
  measurement intact.
- Verified the original live cut with `1118/1118` tests and the complete O1
  foundation repair with `1128/1128` tests.

### Exact K=1–4 continuation and live K=4

- Generalized the layer-major weight-once continuation transaction from K=2
  to exact K=1–4 execution while preserving tokenwise hidden, KV, DeltaNet,
  and native Prefix-Sinkhorn state.
- Added an exact K=4 speculative decoder and causal Qwen3.5 provider with
  mismatch positions 0–3, EOS positions 0–3, verified-prefix restaging,
  terminal tails, mutation rejection, and target-only commit ownership.
- Added a sealed live K=4 harness with an independently executed four-token
  tokenwise control and a same-prompt 2×K=2 control.
- The fixed native Prefix-Sinkhorn trial accepts `4/4` proposals. K=4, 2×K=2,
  and tokenwise control match on output, final target and drafter state, CRSA
  history, and all four position-hidden hashes.
- Against 2×K=2, K=4 target source bytes fall `33.893858%`
  (`151.20 GB → 99.96 GB`), combined target-plus-drafter bytes fall
  `32.391019%` (`158.22 GB → 106.97 GB`), and wall time falls `25.885088%`
  (`261.20 s → 193.58 s`), a `1.349256x` speedup.
- Added a frozen four-prompt long-lived K4 cohort protocol with deterministic
  question-only selection, counterbalanced K4/2×K2 order, exactly-once bundle
  authentication and preflight, complete hidden/state/CRSA parity, atomic
  receipts, and abort-without-retry. The cohort execution remains pending; no
  cohort result is reported.

### Pressure-triggered cyclic GC

- Replaced per-layer forced cyclic collection with a deterministic interval,
  RSS-pressure, and teardown policy. Explicit test collection and teardown
  remain available; metric failures fail closed to collection.
- The fixed causal Qwen3.8 CPU-BF16 A/B is bit-identical across prefix hidden,
  continuation hidden, both state manifests, tokens, cursor, and native
  Prefix-Sinkhorn evidence. Collections fall `134 → 2`, process wall time
  falls `6.875829%`, and speed rises to `1.073835x`.

### Causal Qwen3.5 live drafting

- Extended the exact streamed runtime and flat causal-bundle adoption to the
  pinned Qwen3.5-0.8B profile, including nested text configuration, tied
  embeddings, mixed full-attention/DeltaNet layers, and vision/MTP exclusion.
- Added local transactional K=2 and K=4 draft providers. Draft state advances
  only across target-accepted tokens; mismatch-0, mismatch-1, reset, mutation,
  terminal, and target-only suffix paths remain state-consistent and fail
  closed.
- Proved the causalized drafter's two-token greedy output identical to an
  independent local Transformers BF16 reference on the same pinned checkpoint.
- Added an end-to-end native Prefix-Sinkhorn smoke path in which the local
  Qwen3.5 drafter proposes and Qwen3.8 alone verifies and commits. The sealed
  fixed trial accepts `2/2` proposals and emits the same two target tokens as a
  same-runtime greedy control.
- On that fixed two-token trial, combined target-plus-drafter source bytes fall
  `31.24%` (`151.20 GB → 103.96 GB`) and wall time falls `23.03%`
  (`243.38 s → 187.33 s`), a `1.299x` speedup. This receipt covers one
  fixed prompt and one K=2 verification round.

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
- Generalized the public transaction to the proven `batch=1, K=1–4` tranche;
  larger blocks and batches fail closed. Ordinary decoding is unchanged.
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
- Added a sealed gold-free Q3 K2 benchmark: fixed input projection, two fresh
  causal-bundle runtimes, greedy-versus-speculative token/state parity,
  independently audited staged hidden, authenticated traces, adjusted cost
  receipts, transported validation, and forged bundle/state/cost rejection.
- The official fixed-Q3 draft `[794, 220]` is fully accepted and positive:
  greedy and speculative tokens, committed hidden, cursor, KV, and DeltaNet
  states match exactly. Adjusted source bytes fall `33.89%`
  (`151.21 GB → 99.96 GB`), model time falls `22.44%`
  (`289.19 s → 224.31 s`), forward passes fall `3 → 2`, and head scans
  `2 → 1`; strict provider-state hashing costs `0.228 s`.

### Exact event and binding compiler

- Added a span-aware quantitative event frontend with seventy-four typed
  transaction, rate, comparison, residual, and repeated-duration families to
  exact signed-expression DAGs.
- Added fully grounded exact `Abs` and `Ceil` expression nodes. Both operators
  reject unresolved expressions before lowering, and ceiling-backed capacity
  plans preserve exact rational evidence through the final integer result.
- Added exact calendar-rate binding for fixed 30/31-day months; February and
  multi-month questions remain fail-closed without an explicit day basis.
- Added unique object-possessive binding for original-length relations without
  global pronoun guessing.
- Raised the question-only 64-item public exact-certificate frontier from 3 to
  58; the structural path certifies 57 and the guarded-formula path adds one;
  every other item remains an abstention.
- Upgraded the gold-free abstention audit to partition exact recoveries from
  remaining ambiguity and unsupported grammar. The current report proves 74
  recoveries from the previous 230 abstentions, leaves 156 fail-closed, and
  binds the event/DAG compiler hashes in report provenance.
- Added typed discourse SSA with exact singular/plural referents, role and unit
  identity, topological definitions, target closure, and cycle rejection.
- Added closed calendar/schedule algebra for weekly complements, disjoint day
  sets, explicit periods, frequency conversion, and weekday exceptions.
- Added exact ground recurrence algebra for terminal and cumulative affine
  recurrence and fixed-base growth. The cumulative recovery chain is
  `42 + 5 + 10 + 9 + 4`.
- Added closed bundle/tariff algebra for equal daily budgets, two-day discount
  differences, exact monthly duration ledgers, and typed positive-part
  installation overage charges. The final cumulative chain is
  `42 + 5 + 10 + 9 + 4 + 4`.
- Added typed part, package-capacity, fractional-remainder, pooled-allowance,
  inverse-duration, exact-trip-minimum, temporal block remainder, exhaustive
  unit-rate, recurring-pronoun rate, and absolute weighted-difference DAGs.
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
