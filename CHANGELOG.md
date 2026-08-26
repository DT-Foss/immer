# Changelog

All notable changes to IMMER are recorded here.

## [Unreleased] — 2026-08-26

### Continuous operator harvest, search, and residual discharge

- Extended the real Qwen cartography probe with read-only float64 pre/post
  hidden-state projections. Their arrays are hash-bound to the sealed layer
  evidence and flow directly into the O1 operator harvester; static embedding
  means are absent from this path.
- Connected `scripts/qwen38_o1_cartography.py` to a persistent contextual
  operator bank and graph. Each live probe appends its authenticated Atlas
  measurement, emits contextual transitions, advances a monotonic cursor,
  updates the harvester, and reports every promoted Crystal and edge.
- Added prompt-independent runtime-family identity. Exact per-prompt runtime
  receipts remain intact while code, model/weight revision, dependencies,
  platform, intervention family, feature schema, and action schema determine
  reusable operator groups. Audit coordinates remain provenance instead of
  fragmenting one layer operator into separate groups.
- Added deterministic affine, exact permutation, and row-stochastic Markov
  discovery from contextual arrays with chronological fit/holdout separation.
  Scalar Atlas summaries remain scalar evidence and are never inflated into
  hidden tensors. Historical append-only Atlas heads accumulate continuously;
  foreign heads, placebo arms, inactive measurements, replay, transplant,
  nonlinear holdout failure, and state tamper reject.
- Added NumPy-only constructive Birkhoff-von Neumann decomposition with the
  exact `(n-1)^2+1` component bound, strict theta shape and gauge, O(Kn)
  permutation storage, authenticated reconstruction receipts, and executable
  Crystal publication. The runtime stores only the atoms used by the measured
  operator and applies their weighted outputs with row-vector semantics.
- Added contextual Thompson mutation, diagonal success-step adaptation,
  behavioral Pareto MAP-Elites, matched empirical spectral filtering, and
  basis-invariant Fiedler-projector edge novelty. The fixed operator trial
  stores `33` atoms under the bound `50`, reconstructs within `6.6613e-16`,
  occupies `12` MAP-Elites cells with `14` elites, and learns the successful
  mutation arm on `32/32` late decisions.
- Added verified demand learning for materialized routes: unseen-first UCB1,
  exact negative route/revision/ABI evidence, selection-ordered PPM under
  asynchronous verification, episode-scoped co-occurrence prefetch, and Ricci
  retention `|R| exp(-alpha*age)` with protected pins and full artifact-byte
  accounting. Every public receipt now passes typed semantic reconstruction.
- Added charged-prefix residual execution. A future longer plan selects its
  deepest authenticated charged prefix, executes that stored operator, and
  computes only the primitive suffix. The fixed 128-input trial reuses two
  charged steps, runs two residual steps, releases `9,216` historical work
  units, and matches the four-step primitive path within `1.3323e-15`.
- Closed the imported candidate defects instead of copying them: no silent
  theta padding, no truncated BvN mass, no factorial basis cache, no mislabeled
  CMA-ES, no worse-than-elite replacement, no false sequence learning from
  verifier completion order, no fabricated global episode, and no metadata-only
  byte budget.
- Verified the complete repository with `1,626/1,626` tests under
  `ResourceWarning`-as-error in `410.414 s`. Every changed source, script, and
  test passes Ruff; the changed public files pass compile, JSON, diff,
  link-target, anti-hedge, and private-path gates.

### Compositional crystal intelligence

- Added a finite action-conditioned world model over authenticated one-step
  transition evidence. Distributional planning now composes unseen multi-step
  solutions and gates execution by coverage, entropy, peak probability, model
  revision, and the exact resolved planning policy.
- Added hierarchical Markov options from verified trajectories. Balanced
  Sinkhorn assignment is bound to an exact discrete matching, option kernels
  compose without visited-state bias, and a contraction ledger records the
  primitive depth and accumulated contraction bound.
- Added the generic `ComputeCrystal` VM and content-addressed bank. Typed
  affine, permutation, lookup, and row-stochastic Markov operators execute
  bounded numerical work under canonical schemas, atomic manifests,
  generation/CAS checks, reopen verification, and deterministic recovery.
  Homogeneous affine, permutation, and Markov chains fuse exactly.
- Added the persistent `ComputeOperatorGraph`. It learns only verified one-step
  edges, plans compatible routes, authenticates the full primitive charge basis
  and fusion check, materializes a fused operator, and discharges it later on
  values that did not exist during charging.
- Added algebraic crystallization for additive, multiplicative, and finite
  cyclic invariants. Exact group accumulators execute long compositions after
  score-and-margin admission; ambiguous fits and permuted placebos are rejected.
- Added a continuous PS-Lifted reservoir whose fixed sparse recurrence retains
  delayed state while replicas exchange only ridge sufficient statistics.
  Persistence, duplicate-evidence rejection, barbell/complete topology fusion,
  and shuffled-temporal-label controls are receipt-bound.
- Added calibrated Hopfield novelty energy, explicit regime-change retention,
  and action-conditioned tensor-train/MPO kernels with authenticated error
  recomputation and exact dense fallback.
- Extended the fixed-seed intelligence benchmark. PS-Lifted planning solves
  `250/250` unseen 4–10-step tasks from 68 one-step observations versus
  `226/3,000 = 7.5333%` across all 12 local replicas, `89/250`
  shuffled-action placebo, and `68/250` no-memory. Hierarchical depth falls
  `8 → 1`; topology-shift success remains `100%`. Spectral self-calibration
  cuts the fixed-`pc` barbell consensus from `96` to `58` rounds; the complete
  topology converges in `28`.
- Added the continuous-memory benchmark: fused accuracy is `90.8359%` versus
  `82.1626%` local mean, `50.4532%` no-memory, and `54.2800%` shuffled-label
  placebo on the fixed delayed-state task.
- Added the stored-compute benchmark. A four-edge route charged before its
  inputs existed executes as one operator over 128 unseen vectors, releases
  `27,648` authenticated historical work units, and matches primitive execution
  within `1.7764e-15`. All three algebraic families execute length 12 exactly;
  the disconnected and permuted controls reject.

### Exact ResultCells and live Qwen cohort harness

- Added `ResultCellBinding`, `ColdQwenGenerationReceipt`, `ProvenanceUnit`, the
  atomic `ResultCellBank`, zero-forward `ResultCellExecutor`, and the final
  benchmark layer. Model, code, tokenizer, raw-question hash, rendered-prompt
  hash, token-stream hash, system-prompt hash, generation policy, feature
  evidence, verifier tuple, cold result, and FERTIG judgment form one exact
  binding.
- Derived every teacher-forward baseline from authenticated raw Qwen generation
  evidence. Caller-supplied inflation, mixed provenance, stale identities,
  prompt leakage, resealed tamper, crash boundaries, and forged parity receipts
  fail authentication.
- Added a transient S3/FERTIG executor with a bounded thread-local registry
  keyed by exact feature receipt. It consumes each feature-bound question once,
  persists a prompt-free result document, and reports zero Qwen forwards under
  the exact S3 provenance pair. Without an authenticated cold Qwen-generation
  baseline it records zero saved Qwen forwards.
- Added the frozen real-Qwen cold-then-warm cohort harness with separate
  gold-free `prepare`/`execute` and final `verify` phases. The release cut has
  completed its measurement input: `10/10` local-Qwen cartography jobs,
  `15` authenticated measurements across two sites, Atlas revision `15`, and
  `501.95 s` of model execution.
- Completed the five-cell real-Qwen cohort. Cold generation uses authenticated
  forward counts `[6, 7, 7, 6, 5]`, totaling `31`. The first four rows supply
  exactly four temporal teacher transitions; the fifth holdout transition is
  absent from training.
- Executed the unseen temporal holdout through the learned `mount_organ`
  action. Its authenticated cold baseline is five Qwen forwards; warm execution
  performs zero and commits all `5/5` as saved. Raw Qwen result documents match
  exactly, the final semantic core matches exactly, and the frozen evaluator
  opens once and verifies the gold-correct output.
- Bound the result's certification boundary exactly. FERTIG abstains on the
  holdout in both cold and warm paths, so `fertig_exact_judgment` and
  `fertig_semantic_certified` are false. The quality authority for this result
  is frozen-evaluator correctness plus exact cold/warm parity.
- Confirmed that shuffled-site and shuffled-Crystal controls do not execute,
  both persistent stores audit clean, and exact resume reuses all five durable
  cold ResultCells without repeating Qwen generation.
- Corrected historical Atlas restoration. Scheduler measurement order
  `3, 5, 0, 9, 4` is authenticated as append-history membership while teacher
  transitions retain temporal order `0..3`. Restore now authenticates every
  seen Atlas revision, accepts the initial revision as any authenticated
  member, and requires the current revision to remain the maximum seen head.

### Organism of Experts and direct Qwen/O1 integration

- Added the complete Markov-OoE runtime: calibrated attractor routing,
  distributed Markov-PD agents, fading mobile reservoirs, Möbius rapidity,
  PS-Lifted push-sum, quantized executable Crystals, and immutable
  content-addressed publication.
- Reproduced the original standalone experiment exactly: `96.10%` PS-Lifted
  accuracy versus `17.28%` local, `49.84%` reversible, and `13.50%` shuffled
  placebo; warm Crystal execution reaches `100%`, consensus falls from
  `338` to `64` rounds, and the raw six-kernel payload is `588 B`.
- Bound every learned site to the exact Qwen model pin, `.causal`
  `WeightCoordinate`, action/feature schemas, append-only graph revisions,
  coverage, calibration, verifier hashes, evidence hashes, and consensus
  receipt. Numeric sketches come from real contextual Qwen measurements and
  O1 surprise/progress, never static embedding means.
- Wired the live chain directly:
  `Qwen probe → SemanticWeightAtlas → O1 signal → Markov replicas → PS-Lifted Crystal`.
  Historical Atlas reuse learns without another probe; partial Crystals execute
  covered actions and abstain on every uncovered source.
- Made Atlas history an exact authenticated revision set. Historical
  `(sequence, event_sha256)` membership, concurrent-head changes, rollback,
  forks, stale models, stale weight rails, and forged revisions fail closed.
- Added controller-wide synchronization, execution-bound warm transactions,
  and final `commit_warm`/`reject_warm` accounting. Saved Qwen forwards are
  recorded only after the outer verifier accepts the final result; FERTIG
  mismatch creates zero false savings.
- Added a sealed two-phase promotion protocol and exact forward-drift recovery.
  A crash after any durable Crystal publication resumes from the prepared
  controller state, deterministically finishes the batch, and performs zero
  duplicate Qwen probes.
- Activated native semantic anchors in `Qwen38CausalChat`. Exact-prefix hits
  use the authenticated final-hidden seed; shorter hits evaluate only the
  suffix. Fresh and restored paths match generated tokens, every continuation
  tensor, and final snapshot hashes bit-for-bit.
- Added a verified warm-OoE hook before Qwen in `QwenFertigChat`, while keeping
  FERTIG exact first refusal. Crystal tamper is a hard integrity error; novelty,
  uncovered actions, and missing features fall through once to local Qwen.
- Added direct CompositionRoot and CLI configuration for the local Qwen anchor
  cache. Qwen remains the primary local teacher and novelty fallback.
- Verified the repository with `1367/1367` tests.

### DeepSeek-V4 frontier streaming and live causal crystallization

- Added arbitrary exact suffix continuation after live or restored DeepSeek-V4
  prefixes. Multi-token suffixes execute through the checkpoint's native
  one-token transition, reject overflow before mutation, and clear the whole
  request on an in-flight failure.
- Added native semantic compute-battery anchors for DeepSeek-V4 with deepest
  prefix restore, model-bound final-hidden seeds, snapshot-first publication,
  deterministic byte-bounded LRU, fail-closed empty-target restore, and
  manifest-owned orphan recovery.
- Added one owned local/remote/causal runtime factory and wired it into the PoC,
  layerwise scorer, and benchmark runner. Trace-sparse bundles are rejected by
  general runners; only bundles carrying authenticated full-dense coverage are
  admitted. Benchmark access traces commit before their reports and contain no
  shareable absolute paths.
- Added remote pinned-inventory adoption before trace observation. A live
  immutable checkpoint is header-verified against the causal bundle's pinned
  72,317-tensor layout, then every recorded range is born under that stable
  source identity instead of mutable transport metadata. Inventory SHA and
  fingerprint are bound into benchmark provenance and signatures.
- Made pinned inventory adoption stable across cold and cached header scans.
  When a cache hit has no HTTP file-size field, the scanner derives the exact
  safetensors size from its contiguous validated data offsets; any reported
  mismatch, gap, overlap, coercion, or signed-range violation fails closed.
- Added sealed trace-to-graph ingestion. Multiple traces are identity-checked,
  unioned, reduced to complete six-part expert coordinates, compared against
  current base and append bindings, and handed to the existing crash-safe
  append transaction only for the missing set. Plan mode is offline,
  path-free, canonically hashed, and reports exact payload, staging, resident,
  disk, and cold leaf-transfer requirements before mutation.
- Hardened large expert transactions with block-accurate storage preflight.
  Staged files and sparse target intervals are accounted independently,
  shared and split filesystems receive the correct floors, and both are
  rechecked immediately before the first remote leaf read. Planned recovery
  and completed idempotent replay remain offline and unrestricted.
- Completed the first pinned batched MMLU frontier item. One 44-token CPU-BF16
  forward selected the correct choice B in `2,494.815 s` from
  `50,246,717,453` source bytes with zero rejected charges. Its ingestible
  trace seals `4,754` operations and `8,009` leaves under the causal bundle's
  exact source fingerprint.
- Added an explicit remote-expert split rail for storage-bounded execution.
  The pager reads every dense/control/shared/head/router tensor through the
  local causal tensor reader while every routed expert plan and payload comes
  from the immutable pinned remote Streamer. The mode has no missing-route
  fallback, never touches local sparse expert holes, owns both sources
  transactionally, and records both physical planes and caches separately.
- Sealed the first split-rail replay against the pinned full-remote reference.
  All four MMLU logits and the correct choice B are bit-identical. Item latency
  falls from `2,496.599 s` to `1,772.060 s` (`1.408868x`, `29.021044%`), while
  remote bytes fall by `5,627,335,452` (`11.203862%`). The split reads exactly
  those saved bytes through the local causal dense rail, so total logical
  bytes are conserved exactly with zero fallback and zero rejected charges.
- Wired the same split rail into the resumable layerwise microbatch scorer.
  Split cache capacity and both immutable plane identities are sealed into the
  run identity only when the mode is active, preserving byte-for-byte default
  resume identities while enabling one weight pass across an MMLU cohort.
- Completed the four-item layerwise split cohort in one 43-layer microbatch
  pass. All `4/4` predictions match the sealed full-remote MPS baseline; both
  score `3/4` and fail only the same fourth item. The CPU split run finishes in
  `2,916.11 s` with a sampled peak RSS of `1.46 GiB`. Prediction parity is
  exact across the two device runs; logit identity is evaluated only within a
  single device arithmetic.
- Routed every dense, control, embedding, I64 router, candidate-head, scalar
  head, and batched full-head read through the revision-bound causal tensor
  reader. Expert and dense rails remain separate, missing bindings have no
  fallback, and bounded zero-gap multi-range receipts account for every byte.
- Added crash-safe live expert append. Exact remote leaves are staged and
  hashed before sparse-shard mutation, shard bytes become durable before graph
  visibility, and offline retry reconciles every durable crash boundary.
- Added dense bundle promotion for the exact `1,564`-tensor main-decoder
  contract. The official bundle authenticates `8,845,959,388` dense bytes,
  materializes only the missing `2,136,760,320` bytes, publishes `1,564`
  tensor bindings, and exposes the general dense capability only after complete
  payload and graph verification.
- Transferred the official logical `166.9 GB` checkpoint as a roughly `12 GB`
  physical sparse causal bundle to Beast. A transported 83-token state required
  six new expert rails and then completed all 43 layers locally in `167.764 s`.
  MPS and CPU state are structurally identical but numerically distinct, so the
  transported state remains a proposal. Target adjudication selected the same
  top token on both paths: token `12747`, decoded as the FERTIG-certified exact
  answer `450`; their Top-10 sets are identical with eight equal rank slots.
- Verified the complete repository with `1284/1284` tests, the expanded
  DeepSeek scope with `459/459` tests, and the Qwen causal regression scope
  with `54/54` tests.

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
  demand. Its iso-boundary control requires bit-identical hidden, LM-head, and
  complete serialized native state; a separate best-live one-shot control
  requires the same next token and dtype-bounded drift across the final hidden
  vector and every Attention/DeltaNet/CRSA continuation tensor.
- The first official local CPU-BF16 cell charges a 65-token invariant before
  its 33-token suffix exists. Authenticated discharge cuts the fastest fresh
  path from `130.457495 s` to `91.304167 s`, removing `30.012326%` of peak
  latency for a `1.428823x` speedup. The iso-boundary result is bit-identical;
  all 129 one-shot comparison-state tensors remain inside the pinned BF16
  numerical bound.
- Added a FERTIG verifier that binds one honest replica vote to the complete
  cartography evidence and three bound proof surfaces without presenting
  those surfaces as a fake three-node quorum.
- Verified the complete integration with `1178/1178` tests.

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
