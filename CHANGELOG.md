# Changelog

All notable changes to IMMER are recorded here.

## [Unreleased] — 2026-08-30

### Deployed Markov/MTP product path

- `1e2ba55` added a carried-only `bucket=-1` Beta aggregate. It reconstructs
  algebraically from existing exact carried rows on load; an exact gap bucket
  takes precedence after two observations, while sparse buckets back off to the
  aggregate. Each verified receipt updates its exact and aggregate storage once
  but increments calibration/update metrics only once. On the live SOLARA
  recall, turn 1 used two forwards and 13.75 s. Turn 2 reused 35 Qwen tokens,
  returned the omitted code in ten output tokens, exported a 318 KiB MTP carry,
  and used seven forwards, 34.62 s, and 1.57 GiB peak RSS. Carried updates grew
  `4→7`; the carried `previous=1` aggregate reached Beta(7,1), carried misses
  stayed zero, and all 342 cold updates remained unchanged. This is one fewer
  forward despite one additional output token than the preceding VELORA
  observation. Wall time fluctuated, so no latency claim is attached. Next work
  is higher-position/window learning from carried reliability, not another
  prompt cohort.
- `411744c` separated cold and carried MTP calibration under provider v4,
  hybrid v20, and calibration JSON v2. Beta keys are now
  `(carried, position, bucket, previous)` with independent previous outcomes,
  update counts, and state counts. Legacy v1 rows migrate cold-only into the
  same global persistent file; strict boolean and row-shape validation rejects
  malformed state. A carried row starts at Beta(1,1), so it remains K1 at 0.5
  until ordinary Qwen verification teaches it. With sufficient logit gap, two
  first-ever consecutive hits can open K2; a miss moves carried `previous` to
  zero and collapses the wider path. Warm runtime identities now bind MTP v4
  and hybrid v20. The live VELORA recall retained 35 Qwen tokens and a 146 KiB
  carry on turn 1, using two forwards, 15.13 s, and 1.43 GiB peak RSS. Turn 2
  reused all 35 tokens, returned the omitted code in nine output tokens,
  exported a 314 KiB carry, and used eight forwards, 35.02 s, and 1.59 GiB peak
  RSS. State migrated to provider v4/schema v2: 342 cold updates were untouched,
  carried updates reached four, and all three carried rows were hits.
  Pre-aggregate verification passed 174 tests in 6.979 s; three focused remote
  tests passed. Aggregate review reported no P0–P2. These runs are reported
  separately, not combined into one total.
- `c8c780a` added `Qwen35MtpCarry` v1 across interactive turns. A carry clones
  the exact token history, shifted MTP AttentionState KV, cursor
  `target_cursor - 1`, and last committed target hidden before provider close.
  Its identity binds the relevant model configuration, Q4 manifest, BF16 source
  repository/revision/fingerprint, device, and dtype; finite tensor and shape
  contracts reject mismatches. Export trims terminal speculative overrun to the
  committed Qwen boundary, and import computes only the official suffix.
  Hybrid additionally requires its target-boundary hidden row to bit-match the
  carry. The adapter validates and promotes the carry atomically with the same
  Qwen token prefix; clear, mismatch, error, and close drop both states.
  Markov-only turns materialize MTP lazily only when exporting the carry.
  Calibration remains global and persistent, including v2-to-v3 migration.
  Provider ABIs are MTP v3 and hybrid v19. On the live NIMBUS-60427 recall,
  turn 1 returned `gespeichert`, retained 36 Qwen prefix tokens and a 150.0 KiB
  MTP carry, and used four tokens, two forwards, 13.74 s, and 1.40 GiB peak RSS.
  Turn 2 omitted the code, reused all 36 tokens, returned `NIMBUS-60427`,
  exported a 322.0 KiB carry, and used ten tokens, seven forwards, 31.73 s, and
  1.61 GiB peak RSS. The persistent state migrated to provider v3 with 342
  updates. This is functional exact-carry proof, not a latency improvement: the
  prompt was slower than the preceding Markov-only observation, so the next
  work is carry-context-specific MTP reliability. The pre-review affected suite
  passed 171 tests in 6.553 s; the latest post-fix core runs passed 144 tests;
  nine focused remote tests passed before the live carry run. Final review
  found no remaining P0–P2 issue after validated-pointer and boundary-hidden
  fixes. These are separate runs, not one combined test total.
- `06f7610` turned interactive Qwen context into an in-process compute battery.
  Each successful session retains exactly the committed token tuple represented
  by `model.next_position`; the next official prompt reuses it only when its
  prefix, session ID, model cursor, batch size, and poison state all match.
  Pager-held weights are released while KV/Delta continuation state remains.
  `retain_final_state` stays false, so the uncommitted terminal tail is replayed
  as part of the next suffix without an extra closing forward. Prefix mismatch,
  history eviction, session change, error, abstention, and `/clear` reset the
  binding. One-shot and JSONL execution are unchanged. A restored configured
  hybrid uses Markov-only drafting; explicit restored MTP uses the direct Qwen
  target because a fresh MTP instance has no carried attention state. Qwen is
  still the sole committer. On the same deployed two-turn recall, turn 1
  returned `gespeichert`, retained 35 prefix tokens, and used four generated
  tokens, two forwards, 14.65 s, and 1.43 GiB peak RSS. Turn 2 reused all 35
  tokens and returned `ZORPAX-731` with one prior turn, nine generated tokens,
  three forwards, 18.99 s, and 1.50 GiB peak RSS. The old code path had observed
  four forwards, 29.87 s, and 1.53 GiB on turn 2; this is a before/after product
  observation, not a controlled broad benchmark. The affected local suite
  passed 165 tests in 6.621 s; eight focused remote tests passed in 0.944 s.
  Final review found no P0–P2 issue after the unsafe restored-suffix MTP path was
  rolled back.
- `265d15d` made interactive chat a real multi-turn Qwen session. Every
  successful user/assistant pair is retained and rendered by the official Qwen
  chat template; prompt-bound trimming removes only the oldest complete turns,
  and `/clear` resets the conversation. Contextual turns bypass single-turn
  FERTIG and warm execution even when trimming evicts the complete prior
  history. Single-request and JSONL paths are unchanged. In one loaded raw-Qwen
  hybrid process, turn 1 stored `ZORPAX-731` and returned `gespeichert` in four
  generated tokens, three Qwen forwards, 18.95 s, and 1.51 GiB peak RSS. Turn 2
  omitted the code, reported one prior turn, and returned `ZORPAX-731` in nine
  tokens, four forwards, 29.87 s, and 1.53 GiB peak RSS. This is a functional
  two-turn proof, not a broad performance benchmark. The affected local suite
  passed 161 tests in 6.532 s; six focused deployment tests passed remotely.
- `61cdbaa` restored novelty drafting to the deployed v3 Q4 product path.
  Ordinary chat now runs the round-wise Markov/embedded-MTP hybrid: Markov
  executes learned continuations, MTP covers novelty, and Qwen remains the sole
  verifier and committer. Explicit `--draft-mode markov` disables MTP while
  retaining the v3 target bank. `immer chat --interactive` keeps the runtime
  loaded across arbitrary prompts and provides `/help`, real last-response
  `/stats`, and `/quit`. On the same real prompt with a fixed 24-token cap,
  both modes committed the same intentionally truncated prefix. Explicit
  Markov-only used 23 target forwards, accepted one draft, ran in
  69.469500 s, and peaked at 1,699,160,064 bytes RSS. The deployed hybrid used
  11 forwards, accepted 14 drafts, ran in 49.448516 s, and peaked at
  1,725,218,816 bytes RSS while consuming 52,224 draft bytes and 58,201,088
  target-source bytes. That is 52.17% fewer target forwards and target-source
  bytes, 28.82% less wall time (1.405x), and 1.53% more peak RSS.
- `c204181` added direct native Q4 MLP-page execution behind the explicit
  `--mlp-page-state` option. Exact full-MLP activation traces train bounded
  temporal, cross-layer, and marginal Fixed-Share routes; unready layers retain
  full-Q4 fallback, and continuation transactions commit route learning only
  for accepted rows. Normal chat does not enable this mode, and no wall-time
  result is claimed.
- `6a783e9` replaced greedy Atlas/online continuation selection with a bounded
  beam. Up to eight retained paths combine live Council, Atlas, and calibrated
  prefix evidence; target-only window costs remain authoritative, and beam
  reliability changes only after exact reconciliation.
- `a31bb3d` made direct zero-weight Markov drafting the deployed default while
  retaining the unchanged MTP-capable v3 target bank. Embedded MTP and the
  round-wise hybrid provider now require explicit `--draft-mode mtp` or
  `--draft-mode hybrid`. The same cut routes K1 virtual verification to the
  selected provider, respects request-budget and EOS truncation in verification
  accounting, and accepts authenticated exact-prefix anchor seeds.
- `9f7bec7` replaced target-payload hashing in Markov callbacks with zero-payload
  tensor-version and structural guards. Provider callbacks still fail closed
  on target mutation without reading model-state contents.
- `aeb5d04` made durable Markov state a single write per completed request.
  `close()` skips an already persisted final state, while a loaded migration
  that received no request is still written exactly once.
- `27527a7` added a request-local high-order transition agent. Repeated order-2+
  answer contexts become usable within the same response from confirmed output
  and target corrections; prompt tokens and rejected drafts never train it,
  and the ephemeral agent does not add another persistence write.
- `b080bc0` added a request-local phrase/copy agent. Two or more prior confirmed
  occurrences induce their deterministic longest common continuation prefix,
  expose it through the existing phrase-window policy, and draft up to the
  active K=2–16 ceiling. `0b272ad` bounds both request-local agents to the most
  recent 4,096 confirmed output tokens.
- `1785723` added request-local Council adaptation. Every target-confirmed
  feedback row immediately updates an ephemeral combined Rapidity vector;
  Fixed Share applies the new weights to the next prediction. Finalization
  clears the overlay before replaying each row exactly once into durable
  global and dialect state. Provider ABI is v32 and metrics schema is v23.
- `7ba5116` added request-local recursive-position specialists. A provider-only
  16×8 Beta overlay separates expert skill at proposal positions 0–15 and feeds
  position weights, calibrated confidence, and language evidence before the
  next draft. `182fbf9` covers K1 carry, external mismatch, teacher carry,
  abort, retry, and exact-once persistence. Provider ABI is v33; metrics v24.
- `2caa03f` added same-request lookahead regret. Planned-vs-greedy outcomes are
  tracked per recursive position and can disable a harmful one-step override
  before finalization. `3dff43a` proves terminal-carry persistence failure and
  retry consume the outcome exactly once. Provider ABI is v34; metrics v25.
- `f50aff6` added same-request Surprise/CUSUM regime detection. Abrupt feedback
  shifts update the existing EMA/deviation equations and immediately shrink
  stale combined Rapidities by the canonical regime factor. Durable Surprise,
  CUSUM and regime generation still replay exactly once at finalization.
  Provider ABI is v35; metrics v26.
- `b8b051a` replaced pooled Beam reliability with 16 recursive-position Beta
  rows. Verified prefixes preserve successful shallow positions, the first
  failed position cools itself and its causal tail, position-0 failure still
  forces K1, and external shadow verification trains through its first
  mismatch. Provider ABI is v36; metrics v27.
- `a796f9f` added request-local periodic template inference across changing
  slots. Three aligned cycles can continue invariant future phases even when
  the current exact token context is unseen. `cef25eb` gives exact request
  phrases precedence, requires three phase witnesses, calibrates confidence by
  measured lag agreement, rejects conflicting periods, and gates Atlas phrase
  floors with causal Beam reliability. Provider ABI is v38; metrics v28.
- `1aeef27` added request-local periodic slot binding. A varying future phase
  can copy a confirmed earlier phase when the relation holds across three
  cycles, then continue invariant phases. `b5ee3b8` requires three distinct
  slot witnesses, rejects multiple structural sources even when their current
  token matches, preserves absolute phase through the 4,096-token window, and
  keeps future-block sources untrusted. `898af70` restores global evidence
  ranking across pure/bound periods, permits one bound slot per option, and
  preserves absolute phase without allocating the full answer. Provider ABI is
  v41; metrics v30.

### Arbitrary local Qwen runtime

- Added token-native compositional Markov options. State v6 records each
  target-confirmed prompt/output boundary; Provider ABI v9 deterministically
  derives bounded programs from `Literal(tokens)` and
  `Copy(relative_prompt_span)` atoms instead of persisting duplicate templates.
- Promotion requires at least two distinct confirmed bindings, complete
  support over every compatible episode, two static prompt guards, and copy
  coverage for every variable prompt position. Syntactically ambiguous copy
  programs may coexist only when they materialize one unanimous continuation
  on the concrete new prompt; conflicts abstain to MTP.
- Three ordinary teacher requests produced `CODE_AA11`, `CODE_BB22`, and
  `CODE_CC33`. A fourth unseen `CODE_DD44` request was materialized by the
  Council, target-verified as `CODE_DD44`, loaded no MTP, accepted two
  compositional draft tokens, and completed in three target forwards at
  `1,314,713,600` bytes peak RSS.
- Receipt bootstrap now preserves structured prompt/output identity. A
  prompted `[1,2]→[3,4]` transition no longer collides with a promptless
  `[1,2,3,4]` output; legacy v1–v5 states migrate with unknown boundaries and
  become composition-capable as new confirmed traffic arrives.

- Upgraded the hybrid provider to a one-way per-round cascade. Markov is
  reconsidered on every adaptive wave and can execute any number of learned
  prefixes. Its first K1 decision initializes MTP from the complete
  target-confirmed hidden history and permanently hands off the remaining
  request; no target replay or additional Qwen prefill is required.
- Extended rolling reconciliation with cloned committed hidden rows. K1 sends
  one row and accepted speculative prefixes send exactly their committed
  width; discarded terminal stages send none. Mutation is detected under the
  same target-state isolation guard as provider initialization and proposals.
- A live 24-token continuation used four Markov K4 rounds for the known prefix,
  then one hidden-state handoff and four MTP rounds for the novel suffix. It
  accepted 15 draft tokens, needed only `9` target forwards, and completed in
  `29.9317 s` at `1,456,226,304` bytes peak RSS. The handoff retained 41 hidden
  rows in 419,840 bytes; MTP itself used four head scans and `0.5838 s`.

- Removed the second confidence penalty from target-confirmed dialect phrases.
  Dialect similarity still chooses the applicable memory; once selected, an
  exact suffix match uses its own empirical support instead of multiplying the
  same similarity into probability again. Two confirmed copies of one fresh
  request now open K2 with a seven-token exact option.
- The third ordinary request for that context selected `hybrid → markov`, did
  not construct MTP, accepted all seven proposed phrase tokens, and reduced
  target forwards to `9` for 16 output tokens. The identical output completed
  in `26.7907 s` at `1,341,214,720` bytes peak RSS, versus the first novel
  MTP-backed request at `30.3833 s`, 12 forwards, and 1.49 GB.
- Fused Q4 speculative-row input projections through the existing grouped
  kernel. On each K2 target wave, Full Attention Q/K/V now use one input group
  instead of three and DeltaNet QKV/Z/B/A one instead of four: 176 native
  calls, 352 Q8 row quantizations, and 352 OpenMP team starts disappear across
  the 64-layer stack. Real 27B matrix micro-runs are bit-identical and
  `1.11–1.12x` faster for those projection groups.
- Removed the experimental complete native DeltaNet seam after official-width
  parity caught PyTorch SIMD `softplus` differing from scalar C by one Float32
  ULP. Projection and Conv matched, but recurrent state did not; the product
  continues through the exact composed recurrence. Tiny-topology parity is no
  longer sufficient to activate a native numerical seam.

- Promoted embedded MTP from an explicit mode into the normal deployed chat
  path through a request-locked Markov-first cascade. The zero-weight token
  Council proposes first; a useful K2/K4/K8 continuation keeps the complete
  request on Markov, while K1 abstention lazily opens MTP. The provider cannot
  switch again inside the request.
- MTP fallback feeds the complete target-confirmed episode back into the
  Council, so novel traffic strengthens the cheaper future route. Fixed K2/K3
  tails stay on Markov and never instantiate MTP. Active prefix-anchor caches
  retain their compatible Markov path.
- A fresh ordinary CLI request selected `hybrid → mtp`, generated new German
  text, accepted five draft tokens, and reduced full target forwards from `16`
  to `12`. Generation took `30.3833 s` at `1,492,398,080` bytes peak RSS; the
  MTP provider used eleven physical proposal scans and `1.2700 s`.
- Corrected shared-pager receipts: MTP linears and source bytes are split out
  of the decoder total rather than counted twice. Provider setup, timeout, and
  broken-auxiliary paths retain one complete work receipt. MTP state persists
  before the Council releases its cross-process lock, and banks without all
  embedded MTP matrices fail at runtime load.

- Stopped adaptive embedded-MTP generation at the only frontier the live
  Markov action can consume. K1 now computes the target-known MTP step plus one
  confidence scan; K2 additionally materializes exactly the first proposal
  state. The provider no longer evaluates an unused 15-token tail on every
  round.
- On the same arbitrary 16-token counting request and exact target trace,
  adaptive K2 reduced generation from `34.0514` to `26.7434 s`, target forwards
  from `16` to `9`, and provider head scans from `107` to `8`. Eight draft
  tokens were accepted; provider time was `0.9849 s` and peak RSS remained
  bounded at `1,579,130,880` bytes. This is the first measured Markov/MTP
  speedup over direct local Qwen on previously unknown answer text.
- Split MTP accounting into returned proposal width, physically computed
  proposals, ABI padding, verified proposals, accepted proposals, and actual
  verification rejections. Padded unauthoritative slots no longer masquerade
  as computed or rejected work.

- Enabled F16C half-to-float conversion in the packed Q4/Q8 inner loop on
  capable x86 hosts. Scale values and dot ordering remain identical; other
  architectures retain the scalar IEEE conversion. On Beast, the five-token
  `Fledermaus` generation fell from `18.7056` to `13.4470 s` and TTFT from
  `11.5832` to `7.1037 s`, preserving the exact target trace and ~1.18 GB RSS.
- Repriced Markov K1/K2/K4/K8/K16 actions for the fused F16C backend. A fresh
  persistent MTP run learned through three K1 corrections, opened K2 at a
  `0.667` posterior, accepted its token, and retained the exact target trace.
  Mixed K1/K2 completed in `14.8022 s` versus `14.7884 s` for MTP-K1-only;
  longer high-reliability contexts can now amortize the provider while weak
  contexts remain K1.

- Fused the complete dense Q4 MLP into one native team: Hidden-Q8, Gate/Up,
  Torch-authoritative BF16 SiLU and product rounding, activation-Q8, and Down.
  The dot-product order and packed weights are unchanged. Observer, sparse-MLP,
  FP16, and FP32 configurations retain the composed fallback path.
- Removed the final cross-host numerical ambiguity with a 128 KiB lookup table
  covering all 65,536 BF16 SiLU inputs, generated from the installed Torch
  backend and consumed directly by C. Random multi-row Q4 MLPs match the old
  composition bit-for-bit on Mac and Beast.
- On the identical `Paris` prompt and token trace, generation fell from
  `13.1823` to `12.6874 s` and TTFT from `11.4121` to `10.8587 s`, with peak
  RSS unchanged near `1.176 GB`. The five-token `Fledermaus` trace completed in
  `18.7056 s`; 320 full MLP executions used the fused path.
- Extended the same exact MLP kernel across independent speculative token rows.
  Blind MTP-K2 retained its `2/3` accepted drafts and target trace while falling
  from `21.6381` to `20.1531 s`. The calibrated controller still selects K1 at
  this reliability because direct generation is `18.7056 s`.

- Made explicit Qwen prefix batteries self-charging. On the first miss, the
  runtime derives the question-independent chat-template prefix, executes it
  once, stores native continuation state plus final hidden, resets, and resumes
  the unknown suffix. Markov rolling generation now accepts the same restored
  strict-prefix state instead of forcing a fresh full prompt.
- The deployed empty-system template shares only three tokens. Its real restore
  cost was `0.621 s`, cancelling the saved 192 layer-token evaluations, so it
  is deliberately not enabled by default. Explicit or longer system prefixes
  retain automatic charging. The two-process proof returned fresh answers
  `Paris` and `Rom` with the same 3-token anchor and no answer reuse.

- Activated the embedded Qwen3.5 MTP branch that was already present in the
  official checkpoint. Eight matrices add `238,878,720` Q4 bytes; all 498
  target matrices are hardlink-reused. The 506-tensor bank built in `5.413 s`
  and is bound to the same local causal graph and source identity.
- Added exact EAGLE-style shifted conditioning: `embedding(x[i+1])` pairs with
  final-normalized target hidden `h[i]` at unchanged position `i`. The MTP
  layer reuses IMMER RMSNorm, gated GQA, SwiGLU, shared embedding, bounded
  native LM-head Top-K, and explicit attention state. Target-prefix
  reconciliation publishes only accepted MTP states.
- Added a persistent cost-learning selector over K1/K2/K4/K8/K16. Raw MTP
  Top-2 margins are capped by a first-order Markov/Beta reliability state keyed
  by proposal position, margin bucket, and previous correctness. K1 target
  corrections teach first-token accuracy without opening a speculative stage;
  low-confidence tails stop after one MTP head scan. The target remains the
  sole output authority.
- Real Beast execution preserved output `Fledermaus` and target trace
  `d833f5...b6b81a`. Blind K4 accepted `2/9` drafts and took `25.824 s`; blind
  K2 accepted `2/3` and took `21.638 s`. The calibrated selector chose K1 for
  all rounds, reduced MTP work to `0.593 s`, and completed in `19.855 s` at
  `1.200 GB` peak RSS. MTP remains explicit until learned K2/K4 yield beats the
  measured CPU row cost.

- Removed model-sized process residency from the exact Full-Q4 product path.
  Q4Bank now tracks mappings touched by each already-existing layer execution
  boundary and applies `MADV_DONTNEED` after the synchronous native kernels
  finish. Platforms without mmap advice fall back to unmapping and reopening
  the same authenticated local payload. No weight, activation, dot product, or
  token is approximated.
- Replaced the 1.35 GB all-at-once Q8 LM-head call with one native bounded
  global Top-K kernel. It quantizes Hidden once, preserves BF16 output rounding
  and stable lower-token-ID ties, carries Top-K across row intervals, and drops
  each consumed interval before scanning the next. A 9,001-row multi-interval
  reference test matches the former complete Q8 scan exactly.
- On Beast, the same unseen German request and token trace moved from
  `16,011,718,656` to `1,170,735,104` peak RSS (`-92.69%`). Generation is
  `12.8842 s` versus the former `12.4907 s`; complete one-shot wall time was
  `18.01 s` versus `18.80 s`. An unrestricted eight-token budget stopped on
  EOS after five tokens with the real answer `Fledermaus` while remaining at
  `1,180,954,624` peak RSS.

- Added the first practical packed execution plane for arbitrary local
  Qwen3.8-27B chat. A resumable builder converts all 498 text matrices into
  mmap-native Q4_0/Q8_0 payloads bound to the original repository, revision,
  inventory fingerprint, bundle manifest, layout, and graph revision. The
  original BF16 weights remain untouched and available as fallback.
- Added a package-shipped C ABI with scalar and AVX2 Q4×Q8/Q8×Q8 kernels,
  direct row decoding, activation quantization, OpenMP execution, native-code
  cache identity, and correct FP16 round-to-nearest-even for normal and
  subnormal block scales. Codec ABI 2 rejects the earlier invalid-scale bank.
- Routed ordinary pager linears, multi-row continuations, embeddings,
  candidate rows, and the complete LM-head scan through the packed plane.
  Packed pages use native mmap residency release rather than Python GC.
  Per-request evidence reports TTFT, output tokens/s, Q4 bytes/calls, native
  Top-K and discard counters, physical storage reads, and process peak RSS.
- The real 16-core AVX2 run reduced the 53.79 GB text source to a 22.03 GB
  packed plane and produced readable arbitrary German output. The first
  eight-token result completed in 73.21 seconds with 22.58-second TTFT and
  21.65 GB peak RSS; this is the new runtime baseline, not the finish line.
- Added backend-aware K1 abstention to the Markov per-wave controller. Direct
  execution competes with K4/K8/K16 under the same expected-token/work
  equation. On the mmap Q4 CPU plane, `C(K)=1+0.9(K-1)` reflects neural-row
  compute after transport has already vanished. Low-confidence waves consume
  state directly without a transactional clone, still receive target feedback,
  and persist the confirmed episode. The real Meerwasser request used seven K1
  waves, zero draft bytes, preserved the direct token trace, and completed in
  74.58 seconds versus 79.05 seconds direct. Local Qwen3.5 accepted five draft
  tokens but took 112.97 seconds, so it is not the CPU product provider.
- Packed payload acceptance now uses a stat-bound SHA cache: the builder knows
  every digest while writing, normal 498-file mount takes 0.046 seconds, and a
  changed inode/size/mtime/ctime triggers a full digest check. Repeated Q4 rows
  decode once and restore caller order. Transient Q4 input, activation, output,
  row-restore, and head-sort workspaces obey the existing resident-memory cap.
- Added native same-input projection groups. Q/K/V, DeltaNet QKV/Z/B/A, and
  MLP Gate/Up share one Q8 activation quantization and one native thread team
  while preserving separate tensor payloads, outputs, logical call accounting,
  and the exact generated token trace. The full eight-token wall time remained
  neutral; the grouped counters make that result explicit instead of claiming
  a false speedup.
- `immer chat --jsonl` now keeps the verified bundle, tokenizer, model, native
  kernel, and 498 mmap tensors alive across raw-text or `{id,message}` lines.
  The real two-request process generated first tokens in 15.01 and 16.90
  seconds; request two opened zero new tensors. `--max-requests` exits directly
  after the configured line count.
- Replaced PyTorch's generic 10,240-group `conv1d` in the fixed Qwen DeltaNet
  kernel with a direct causal small-kernel implementation. K=4 windows are
  expressed as shifted channelwise products with FP32 accumulation; raw Conv
  state and BF16 outputs remain bit-identical. On Beast, S=1 improved 162.8x
  and S=18 improved 47.5x. The same arbitrary eight-token request fell from
  79.05 to 23.01 seconds generation, TTFT from 25.05 to 13.65 seconds, and
  preserved the exact token trace.
- Promoted the recurrent precision bank as the local default. MLP Down moves
  from Q8 to Q4 while recurrent Linear-Attention, embedding, and head remain
  Q8. The derived plane falls from 22.03 to 19.18 GB and real peak RSS from
  21.65 to 18.79 GB. After warm pages, the same trace completes in 20.50
  seconds with 12.10-second TTFT. The superseded Balanced directory was removed
  only after 434 shared hardlink inodes were verified; original BF16 weights
  remain unchanged.
- Promoted the complete Q4 projection bank after both known arbitrary German
  prompts preserved their prior token traces exactly. Only embedding and LM
  head remain Q8. Payload falls again from 19.18 to 16.40 GB and peak RSS to
  16.02 GB; the Himmel request completes in 20.12 seconds with 11.95-second
  TTFT. The recurrent directory was removed only after 258 shared hardlinks
  were verified; its 240 superseded Q8 Linear-Attention files were the only
  payloads released.
- Added a read-only receipt importer for persistent Markov memory. It extracts
  deduplicated prompt+generation token episodes from explicit JSON roots,
  resumes episode-by-episode, and commits each receipt digest in the same
  atomic state transition as its learned episode. Digests survive bounded
  history eviction, eliminating the former state/journal crash window. Legacy
  journals migrate read-only into state v5; dry-run performs no writes and
  source files remain unchanged. Beast imported the complete 40-token German
  answer with zero model forwards; a second run imported zero. State now holds
  869 tokens, nine episodes, four dialects, and nine updates.

- Generalized correction-first rolling verification from fixed K=4 to a
  configurable K=2–16 target window; `immer chat` defaults to K=8. All target
  positions remain layer-major, so one checkpoint-matrix read serves the whole
  wave. Full K16 acceptance emits 16 exact greedy tokens with one target stack
  pass after prefill.
- DeltaNet prefix traces now retain the compact Conv inputs that fall out of
  the final rolling kernel state. Any accepted prefix through K16 reconstructs
  exact Conv and recurrent state without a replaying weight pass; K<=kernel
  windows retain no extra Conv-prefix copy.
- Qwen3.5 transactional drafting and the zero-model-byte Markov Council now
  emit variable tails up to 15 tokens. Reconciliation advances only through
  the target-confirmed prefix. Repeated global or dialect episodes now produce
  the longest supported option up to 15 tokens instead of a fixed triple; all
  option tokens remain ordinary target-trained Council rows.
- Added an opt-in persistent Markov controller for the real rolling window.
  It chooses K4/K8/K16 below the configured ceiling from bounded contextual
  token sketches and terminal target receipts, optimizing accepted drafts per
  target/draft/aux bytes, forwards, and time. Cold traffic stays at K8; zero
  acceptance, errors, and timeouts move later arbitrary chats toward cheaper
  windows without any calibration prompt or additional model execution.
- The window controller now executes a real policy: K8 cold start, one
  bootstrap observation for each allowed arm, then sampled Fixed Share with
  the true action propensity. Wider verified waves teach exact shorter-prefix
  reliability through a separate acceptance prior without inventing unobserved
  work cost; that evidence also satisfies shorter-arm bootstrap. Council
  confidence, disagreement, and variable phrase strength shape the next
  horizon. State is bound to the
  target, tokenizer, and provider identity; K2/K3 retain rolling drafting; a
  post-generation infrastructure fault no longer poisons window learning.
  Target timeouts persist their elapsed work and source-byte delta as negative
  feedback, preventing an unobserved long arm from being bootstrapped forever.
- Markov request windows are now hard ceilings around a per-wave selector.
  One zero-weight Council call emits the maximum tail plus prefix-local
  confidence, disagreement, and phrase evidence. Expected accepted-prefix
  utility chooses K4/K8/K16 before target staging; only the selected prefix is
  scanned, while the unused provider suffix remains explicitly unauthoritative.
  Mixed-window generation receipts bind request ceiling, actual K, provider
  tail, staged prefix, selector inputs, and accepted prefix for every round.
- Added persistent Markov intelligence below the model pagers. Streamer access
  events become grouped operation states with tensor/read-kind tags; order-1
  and order-2 experts learn next-operation distributions with Fixed Share,
  surprise/CUSUM regimes, and Ricci-bounded node/context retention. Confident
  predictions issue bounded Linux `POSIX_FADV_WILLNEED` hints for exact local
  ranges without producing payload bytes, observer recursion, budget charges,
  tensor residency, or a second weight cache. CLI exposes state, confidence,
  support, and per-operation byte bounds. Source identity is validated before
  observer attachment, which occurs after every one-time runtime/preflight read.
- Expanded range prediction into a multi-step Markov reservoir. A bounded
  depth/width beam rolls stored order-1/2 distributions without learning from
  hypothetical states. Cumulative path probability, normalized Ricci value,
  predicted reuse distance, and forecast distance rank future operations.
  Leaves are deduplicated only after operation ranking, distributed across one
  global byte/leaf budget, and protected from repeated hints by an operation
  cooldown. Metrics expose candidates, paths, reservoir truncation/dedupe,
  exact hinted leaves, and hit/miss coverage separately for each distance.
- Closed the range-prefetch learning loop. Kernel-accepted hints remain delayed
  actions until later exact demand settles byte-overlap utility. Persistent
  d1–d8 rapidities learn from `2*useful/hinted-1` with a Fixed-Share floor and
  reweight future beam scores. A settled utility EMA scales the effective
  reservoir from 12.5% to 100% of the configured hard byte maximum. Early hits
  count, due misses settle negative, declined/error hints consume neither
  feedback nor cooldown, and v1 range state migrates to distance-agent v2.
- Expanded the exact residual-PQ head query union from four to sixteen rows.
  Certified pruning and stable top-k parity now cover the complete rolling
  window; the weight-only builder records the same K16 scorer bound.
- Added a second exact pruning tier inside surviving head leaves. It evaluates
  each vocabulary row's existing PQ code tuple plus residual radius, unions the
  possible winners across K<=16 queries, and direct-fills only those checkpoint
  rows. Selected rows are zero-filled into the canonical page shape before the
  unchanged FP32/BF16 scorer, preserving kernel shape and stable ties. A leaf
  with no certified byte saving uses the former contiguous read once. A
  transport cost gate also requires at least 50% row removal and at most four
  survivor runs, preventing small logical savings from exploding into tens of
  thousands of tiny local reads. Row-cap evaluation itself stops after 64 leaf
  probes when none produces a physical saving.
- Added a complete prompt-free Fast-MLP builder for all 64 Qwen layers. It
  computes Gate/Up Gaussian joint moments directly from local causal weights,
  constructs deterministic p4/k32 pilots, and publishes the existing packed
  pilot plus Down-transpose mount ABI without a tokenizer or model forward.
- Added bounded online Fast-MLP routing. The unchanged exact MLP path feeds its
  already-computed Gate, Up, activation, and output tensors into a fixed-size
  recursive-ridge controller. Cold, low-capture, wide, and periodically due
  routes stay exact; confirmed routes use the sparse executor. Optional state
  persistence is locked, atomic, plan-bound, and contains no prompt or hidden
  tensor content. All 64 layer updates flush once at the request boundary
  instead of rewriting the fixed-size state after every layer.
- Added `qwen38_fast_mlp_existing_evidence.py`. It reuses authenticated
  Gate/Up/Output tensors already stored by O1, reads each layer's exact Down
  weight once, and writes a separate calibrated candidate state with no prompt
  or model forward. Persisted evidence budgets are reopened exactly; failures
  remove the candidate and never overwrite the input state.
- Fast-MLP width is now an exact-confirmed action instead of a fixed 32-block
  constant. Exact paths measure the worst-row cumulative capture curve and
  admit the minimum width reaching `0.50`, up to 128 blocks; layers missing the
  floor stay full. Later waves widen for current pilot-score mass, reuse the
  existing 13.55 GB payload unchanged, and migrate v1 online state in place.
- Capture-only Fast-MLP admission is disabled. Weight-only layers now require
  prequential sparse-Down shadow calibration with recent worst cosine
  `>=0.999`, relative L2 `<=0.05`, and a persisted scalar correction; v1/v2
  states migrate output-unconfirmed. A pre-dynamic-read economics guard also
  returns to the full MLP when combined sparse row transport exceeds `90%`.
  Shadow calibration masks one worst-row activation and reuses the exact Down
  matrix while resident, adding no checkpoint or auxiliary source read. A
  maximum-width capture miss skips output shadowing entirely.
- Added a deterministic all-layer p4/k32 Fast-MLP builder over authenticated
  local weights. Isotropic Gate/Up joint moments seed all 64 routers; local
  transpose and packed-pilot banks replace the `34.23 GB` full MLP target wave
  with an `18.0147%` single/reused-route row footprint without prompts, model
  forwards, cartography, holdout data, or Hugging Face.
- Added bounded target-confirmed Fast-MLP learning. Exact paths feed their
  already-materialized Gate/Up/activation/output tensors into fixed-size
  recursive-ridge state, while low confidence, warmup, width mismatches, and
  periodic checks request the unchanged full MLP. Optional state is locked,
  SHA-bound to its plan, atomic, and content-free.

- `immer chat` now opens the authenticated local Qwen3.8 causal bundle
  directly. General chat no longer depends on an unrelated S3 or FERTIG
  composition manifest.
- Mounted the local Qwen3.5-0.8B model as a K=4 draft provider for arbitrary
  chat input. The target executes four proposed token rows layer-major while
  reading each checkpoint matrix once and commits only target-confirmed text.
- One-shot generation no longer streams the complete 64-layer target stack to
  materialize a final continuation state that the chat adapter immediately
  destroys. Terminal K=4 EOS and fourth-token correction paths likewise
  return their already verified text without replaying the confirmed block.
- Target, drafter, and combined source bytes now remain separate in runtime
  evidence. Target forward counts and target byte counts describe the same
  execution boundary.
- CPU BF16 linear execution now aliases authenticated range buffers directly
  instead of cloning every streamed matrix. Public tensor reads retain their
  copy-owning contract.
- Added the production fast-MLP mount for arbitrary chat. One artifact root
  loads the existing router, affine correction, packed pilots, and row-addressed
  down-projection transposes; the target pager stays shared, auxiliary traffic
  is accounted separately, and K=4 automatically executes through
  `execute_many()` on every selected layer. The real cached Beast bundle plus
  the existing layer `0,63` fast banks mount in `2.274 s` with no model
  forward.
- Local chat defaults now cover a complete 64-token request instead of
  exhausting the cumulative range-read budget after the first target sweeps.
  Target residency defaults to `192 MiB`; the drafter and fast-MLP auxiliaries
  each default to `64 MiB` without allocating those limits up front.
- Complete local bundle verification now memoizes each fully hashed shard by
  manifest identity plus device, inode, size, mtime, and ctime. Unchanged
  55.6 GB checkpoints reuse their authenticated digests on later chat starts;
  any shard identity change forces the original full-content hash again. On
  the official Beast bundle, cold verification takes `77.772 s`; the unchanged
  next-process reopen takes `1.198 s`, a `64.898x` startup speedup.
- Replaced replay-based K=4 chat with a correction-first rolling wave: one
  target-known token leads three drafts, the target executes all four rows
  once, and any accepted prefix commits from a compact DeltaNet update trace.
  Prefix widths `1..4`, a following decode, Graft history, and native
  Prefix-Sinkhorn usage are bit-exact while partial commit performs zero
  checkpoint reads and zero linear projections.
- Added native Qwen-token Markov drafting. Sparse variable-order counts operate
  directly over token IDs, require no sibling model and no draft weight bytes,
  learn only target-confirmed prefixes, persist bounded cross-request memory,
  and remain subordinate to rolling target verification.
- Expanded Markov drafting into an eight-agent council spanning orders
  `0/0/1/1/2/4/8/16` and windows `128..4096`. Linear probability pooling is
  updated by target-only log-likelihood advantage, Rapidity EWMA, Fixed Share,
  and surprise CUSUM. A deterministic regime test moves structured experts to
  `24.37%` each, then shifts the two unigram agents to `48.125%` each while
  preserving a `0.625%` floor for every dormant expert.
- Persistent Markov memory now stores complete request episodes exactly once,
  injects non-proposable episode boundaries during fitting, migrates v1 state,
  records expert accuracy/disagreement/effective count/regime generation, and
  holds a POSIX state lock through each request so concurrent writers cannot
  lose confirmed learning. Expert feedback and episode bytes are deferred to
  one terminal transaction; an aborted half-request persists neither. State
  reads use a stable `O_NOFOLLOW` descriptor and reject inode/time replacement.
- Added Council dialect memory v3. Bottom-k token N-gram sketches select up to
  64 context profiles without storing raw context in the profile. Similar
  prompts reuse a profile (`0.8125` under a one-token perturbation), unrelated
  prompts split, and opposite profiles recall opposite expert regimes even
  after global weights adapt elsewhere. Dialect force scales with similarity;
  one provider is explicitly one request. At capacity, Ricci retention
  `visits * exp(-0.001 * age)` evicts the least useful stale profile.
- Markov state now migrates both v1 and Council-v2 into dialect-v3. Dialect
  profile, global Council update, regime state, episode boundary, and confirmed
  tokens share the same terminal atomic write and process lock.
- Added target-confirmed phrase-option agents above the native-token Council.
  Completed episodes produce variable continuations up to 15 tokens after
  repeated support: two matching episodes inside the active dialect or three
  globally. Options never cross episode boundaries and never bypass Qwen
  verification.
- Markov state v4 binds every completed episode to its active dialect and
  migrates v1, v2, and v3 state in place. Dialect and global phrase candidates
  compete by confidence, support, matched context depth, and dialect
  similarity; a weak local phrase cannot override stronger global evidence.
- Phrase tokens are teacher-forced through the ordinary Council prediction
  rows, so accepted and rejected prefixes update the same target-only expert
  feedback as ordinary Markov drafts. Runtime evidence reports proposed and
  accepted phrase tokens, source, support, and confidence.
- Ordinary committed decode now transfers continuation-cache ownership one
  layer at a time. Each old KV or DeltaNet state is released immediately after
  its replacement is produced instead of retaining complete old and new
  64-layer stacks until final norm. For the official topology, the removed
  duplicate cache is `154,927,104 + 65,552 × prefix_tokens` bytes.
- Fast-MLP K-token execution now unions overlapping dynamic Gate and Up rows
  across the complete wave, reads each target neuron once in one-route-bounded
  chunks, restores every row's original neuron order, and retains the existing
  per-row down-projection arithmetic. Identical K4 routes remove `75%` of
  dynamic Gate/Up target transport; disjoint routes read exactly the former
  byte count. Runtime trace v2 records requested rows, unique rows, and reuse.
- Fast-MLP runtime trace v3 adds a one-route down-transpose block cache.
  Up to 12 unique routes are ordered by exact bitmask-DP block-reload cost;
  larger generic batches use deterministic maximum-overlap greedy ordering.
  Every row's original block, activation, and reduction order is restored
  before multiplication. Identical K4 routes remove a further `60 MiB` per
  active p4/k32 layer; disjoint routes read the former byte count.
- Added stable direct-fill local transport. `LocalRangeReader` uses `preadv` to
  write an exact inode-validated range into caller-owned writable storage,
  retries after replacement, handles progressing short reads, preserves cache,
  budget, and access-trace accounting, and falls back to `pread` only when the
  platform lacks `preadv`.
- CPU Qwen pagers now allocate the final BF16/F32 Torch weight tensor first and
  fill it directly through the CausalTensorReader plan. Sorted unique selected
  rows are written into their final row tensor run by run, eliminating both
  the intermediate Python body and the later stack duplicate. Unordered or
  repeated IDs retain one bounded restore gather. Transport policy is
  `local-range-direct-fill/v2`; pager policy is
  `one-shot-qwen35-direct-fill/v4`.
- Added the exact causal LM-head rail. A deterministic weight-only builder
  trains product-quantized subspace codebooks, stores per-row outward residual
  radii, and builds a best-first tree of code-presence masks over canonical
  token pages. A 160-subspace/256-code/64-row design is about 67 MB
  for the official 2.54 GB head.
- Mounted head search uses an explicit scorer ABI: FP32 accumulation followed
  by BF16 round-to-nearest-even. Residual, LUT, norm, accumulation, underflow,
  and output-rounding bounds are outward; BF16 subnormals are rejected by raw
  bit inspection; possible overflow produces an infinite, never-prunable cap.
  Internal nodes prune only when every K<=4 query is certified, and stable ties
  retain the lower token ID.
- Unproved leaves run the ordinary canonical page read and score exactly once.
  Loose certificates therefore degrade to one complete head scan without
  duplicate page reads. Missing indexes preserve the legacy backend scorer;
  valid but non-applicable indexes detach and use the full scan, while corrupt
  or cross-model artifacts fail during mount.
- Added `qwen38_exact_head_build.py`, `immer chat --exact-head`, a bounded
  artifact mount, request-level pruning evidence, deterministic serialization,
  dirfd/O_NOFOLLOW loading, source-identity bracketing, and structural tree,
  mask, radius, hash, and payload validation.
- The explicit Fast-MLP mode now packs the independent K-token projection rows
  into one physical GEMM per matrix. Exact mode retains the original separate
  kernels; packed execution is bound into the Fast runtime identity and
  reports physical packed-call and row counts.

### Receipt-native Seed v3 and O1 information geometry

- Integrated the shared Seed v0.3 architecture as a native IMMER shadow
  proposer: GRU recurrence, trainable Causal Prefix Sinkhorn Attention,
  joint Gate×Up microkeys, contextual operator and novelty heads, predictive
  and quotient heads, plus route-value and expected-work heads. The same core
  now accepts either tokens or the existing 64-dimensional authenticated
  Qwen/O1 receipt sketch.
- Added the sealed receipt-to-training ABI. Measurement, execution-learning,
  and demand-execution receipts retain their model, code, site, graph,
  authority, verifier, group, generation, and split identities while exposing
  aligned `[group, sequence, feature]` tensors. Train, calibration, and
  holdout groups cannot overlap. Promotion risk counts admitted groups instead
  of correlated layer rows.
- Added deterministic D-optimal O1 acquisition. A structural design derived
  from immutable `ProbeJob` fields selects a bounded information-maximizing
  frontier subset through exact replayable log-determinant gains; prompt
  identity, layer topology, intervention, module, and unit structure are
  included, while outcomes and holdout values are rejected as inputs.
- Added exact modular-affine discovery over explicit prime fields. A bounded
  deterministic Gauss-Jordan fit identifies `y = Ax + b`, verifies every
  training and chronologically separate calibration transition, and emits the
  existing `AffineActionAtom` runtime type with a sealed replay receipt.
- Added SHA-first Seed inference checkpoints and a strict v0.3 migration path.
  The trained Micro and 5M checkpoints now load without optimizer/RNG payloads;
  their hidden states, LM logits, operator/state/final/novelty/predictive heads,
  and MLP microkeys remain tensor-identical to the source models on fresh
  contextual programs. Migrated manifests are `9fa2e6d3…` and `8e88c0d0…`.
- Closed the complete spawn-safe warning-fatal repository gate at
  `1,951/1,951`, `OK`, `785.153 s`, exit `0`; the post-review Seed regression
  gate closes `9/9` and the final affected matrix closes `114/114`. The built
  `immer-0.8.0` wheel contains every new runtime module and hashes to
  `9440a8c0…`.

### Receipt-bound Qwen MLP layer governor

- Rebuilt the compute-battery formulas directly against IMMER's own receipts
  as a brake-only layer policy: it may remove already measured sparse layers,
  never add or promote one.
- Extended the mounted Qwen comparison from assistant onset to a four-token,
  teacher-forced trajectory from one exact native prefix. Running all eight
  measured sparse layers preserves Top-1 on `3/4` steps, with mean Top-10
  overlap `8.75/10`, minimum hidden cosine `0.922312`, and maximum relative L2
  `0.396926`. The single Top-1 break is now a first-class rejection signal,
  not an averaged-away metric.
- Added a receipt-bound layer-budget selector over complete decode reports.
  The selected brake leaves only layers `0` and `63` sparse and reaches Top-1
  parity `4/4`, mean Top-10 overlap `9.25/10`, minimum hidden cosine
  `0.983171`, and maximum relative L2 `0.182834` while saving `1.8003%` of
  model-weight transport. Policy file SHA
  `ab6ae5ebf1f7f60c528025cd1fe43efe714b7f8c9c218682dc8c407be6f93e09`
  is bound to the exact prefix, token path, model, router, affine correction,
  verifier, and four-step horizon.

### Full-span O1 authority and all-layer MLP evidence v2

- Added a dedicated passive `full-span-v2` O1 probe family. Full-span jobs
  always execute their own Qwen measurement; a matching layer-63 coordinate
  can no longer reuse an older one-layer result and impersonate a 64-layer
  execution.
- Closed frontier generation `5`: ten full-span measurements over two
  chronological prompt generations produce exactly `640` layer-local
  input-to-output context receipts, while the scheduler closes `110/110`
  jobs. The Harvester and CrystalStore now have a preflighted `256 MiB` state
  envelope for this prompt-by-layer authority map.
- Added `CaptureManifestV2` and a durable `PLAN.json`. Its prompt-major plan is
  train `3`, calibration `2`, holdout `5` across every Qwen MLP layer, for
  exact split targets `192/128/320`. Bank recovery replays receipts before
  opening selected tensor payloads, enforces the train-to-calibration-to-
  holdout transition, and binds every authority to its exact job and probe
  family.
- Added a 64-layer calibration lock and layerwise analysis. Every layer seals
  its own `3/2` prefix corpus and fit before the later-generation holdout can
  enter the bank; restart and analysis preserve the manifest plan and can
  restore only the requested split and layer payloads.
- Completed the full bank at `640/640`, exact `192/128/320`, audit-clean,
  with `6,118,965,248` referenced tensor bytes and complete state
  `5d91c53e…`. All 64 layer holdouts return the same `146/549` exact hits.
  Of `1,389` mechanism-local candidate admissions, `1,370` are metrically
  identical to their matched random-coordinate control; the other `19` only
  avoid random collisions on the same uniform template hits. No content
  runtime path is promoted from this all-layer exact-key line.

### Markov coordinate intelligence

- Added a sealed Markov coordinate-selector tier over exact Qwen joint
  gate/up evidence. A basis is the state, adding one coordinate is the action,
  and exact chronological collision reuse is the reward. Train-only Energy,
  Fisher, variance, and deterministic-random nominators feed a bounded beam;
  calibration locks one model before holdout.
- Added the nominal collision-entropy gate `2 * k * quant_bits >= 64`. It
  blocks compact `k=1` states that fit train/calibration but remain exposed to
  birthday collisions, while retaining exact collision verification as the
  promotion authority.
- Split holdout reporting into frozen-cache and online-adaptive metrics and
  added equal-`k` Energy/Random plus Full/Marginal controls. The selector opens
  an MLP bank with holdout tensor payloads deferred, persists its fit, then
  performs a full payload audit before reading holdout tensors.
- Added a deterministic label-free unseen-prompt cohort builder. Selection is
  derived only from item identity, question hashes, a pinned seed, tokenizer,
  and prior prompt identities; gold answers, model outcomes, and benchmark
  status cannot affect the selected registry.
- Added sealed prompt-row roles derived from the exact token registry. The
  longest shared chat-template prefix and suffix are removed before selector
  fitting, and each content-row selection is bound back into the projection
  evidence.
- Added the coordinate-independent exact-output reuse ceiling. On the first
  official bank the content-only ceiling is zero: the earlier adaptive
  `116/512` L54 hits are shared chat-template rows, not question-content reuse.
  Content mode now persists that result and exits before fitting or opening
  holdout tensors when no exact-output reuse signal exists.

### Layer-local Qwen MLP pilot agent

- Added a sealed layer-local Gate×Up range router. Each 64-neuron block is an
  action; residual OMP chooses four train-only pilot neurons per block, and a
  tiny ridge kernel maps their current SwiGLU energy to the next sparse weight
  ranges. A deterministic random-pilot policy, equal-compute static marginal,
  and per-row oracle are evaluated under the same neuron budget.
- Completed O1 frontier generation 3 at `90/90`, then captured a second clean
  `40/40` exact Qwen3.8-27B MLP bank from five label-free prompts absent from
  generation 1. The frozen p4/k32 agent touches `3,008/17,408` neurons
  (`17.279%`). Calibration captures `41.425%` activation energy versus
  `27.755%` static; the external generation captures `39.466%` versus
  `27.754%` static and `25.890%` random over `2,832` content rows.
- Added the executable BF16 two-pass kernel: resident pilot rows execute first,
  then non-pilot rows from the 32 selected blocks are added. Across all
  `14,499,840` external down-projection values, every full reference output
  reconstructs bit-exactly. The sparse agent reaches cosine `0.8790` and
  relative L2 error `0.4995`, versus static `0.7590` and `0.6947`; output error
  falls `28.10%`. The receipt remains `promoted=false` until its residual
  correction closes the remaining output gap.
- Added the train-only diagonal affine residual Crystal. Generation 1 fits one
  scale and bias per output coordinate (`10,240` scalars per layer), persists
  the complete `860 KB` fit, and only then opens generation 2. External cosine
  rises `0.8790→0.94524`, relative L2 falls `0.4995→0.32638`, and reconstructed
  output energy rises `53.29%→89.49%`. Against equal-compute static, total L2
  error falls `0.69474→0.32638` (`53.02%`). Runtime correction costs one
  elementwise multiply and add; promotion still requires later-generation
  logit and answer parity.
- Added true selected-row execution to the Qwen pager and mounted the sparse
  executor directly in `StreamedQwen38` continuation layers. A local
  `1.426 GB` down-projection transpose bank makes columns row-addressable; a
  `267 MB` packed pilot bank collapses fixed pilots to three reads. Prefill,
  unmeasured layers, and Gate/Up/Activated observation automatically retain the
  exact full MLP. Sparse fit identity is bound into snapshots and staged
  continuations.
- Verified physical execution on all eight measured 27B layers. The mounted
  path reads `18.015%` of MLP weight bytes and every layer is faster:
  `1.31×–1.94×`; aggregate time falls `3.419→2.289 s` (`1.494×`) and transport
  falls `4.278 GB→770.7 MB`. Consolidation reduces original Causal range reads
  from `2,366` on the first sparse layout to `48–58` per layer, plus three
  packed-pilot and 24–29 transpose reads.
- Ran the mounted path end-to-end from the same exact native prefix state on
  all five unseen prompts. Assistant-onset Top-1 is identical `5/5`, mean
  Top-10 overlap is `8.4/10`, final-hidden cosine averages `0.9466`, and mean
  relative L2 is `0.3276`. Decode transport falls `243.53 GB→225.99 GB`
  (`92.80%`); wall time is effectively neutral at current `8/64` coverage
  (`320.06→318.58 s`, `1.0047×`). Status remains unpromoted until multi-token
  and verified-answer parity replace the assistant-onset gate.
- Added a spawn-safe warning-fatal unittest runner that removes Python 3.13's
  SentencePiece SWIG capsule before interpreter teardown. The complete suite
  closes `1,899/1,899`, `OK`, `740.286 s`, exit `0`; post-mount affected-file
  coverage and the final process/runtime gate also close cleanly.

### Live exact Qwen MLP capture runtime

- Mounted `LiveExactMlpCaptureRunner` directly on the local causal Qwen runtime
  and IMMER's native attention path. A fixed layer allowlist captures the four
  ordered MLP boundaries from one partial forward per prompt and split. The
  canonical `25/10/5` plan therefore uses five prompt forwards per split and
  `645` total layer executions instead of one full-model forward per cell.
- Added split-closed execution: all five prompts for a layer publish as one
  atomic resume unit, calibration seals the `25/10` model inventory before the
  five holdout groups can run, and fixture evidence remains outside the live
  corpus. Eight selected layers use exactly `24` weight-local `linear_many`
  replay reads across gate, up, and down projections.
- Bound all 40 prompt/layer cells to exact O1 scheduler, Atlas measurement, and
  Harvester authorities. One manifest, ModelPin, and verifier pin govern the
  bank; the bank invokes the concrete replay verifier and publishes its proof
  through CAS. Exact source-dtype array bits live in compact content-addressed
  objects with crash recovery and orphan auditing.
- Completed the official local Qwen3.8-27B run: `40/40` groups, exact
  `25/10/5` closure, `audit_clean=true`, and `369,098,752` referenced tensor
  bytes. The minimal promoted candidate uses joint gate/up coordinates
  `[6844,3028]` at `k=2`, 16-bit quantization: calibration `232/1024` exact
  verified all-row hits and holdout `116/512`, both with zero wrong collisions.
- That candidate stores `126,760 B`, versus `221,043,712 B` for the full
  17,408-coordinate 16-bit control with identical held hit/miss counts: a
  `1,743.80x` reduction. The layer-54 holdout is a topology holdout over the
  same five prompts. A deterministic random `k=2` control also reaches zero
  wrong collisions at 12 bits. Content-row decomposition subsequently proves
  zero exact output reuse, so this result is a compact collision-safe key for
  template repetition, not a content-compute saving mechanism.
### O1 exact measurement sidecars

- Added an opt-in native observer for the exact
  `streaming_prefix_log.routed` tensor before Head-CRSA blending. It preserves
  absolute tokenwise query/key positions, copies at most 32 Prefix-Sinkhorn
  rows into detached readonly CPU `float64`, emits nothing for `alpha=0`, and
  adds no tensor operation or copy when disabled.
- Added bounded Prefix-Sinkhorn capture to Qwen cartography and O1. Raw matrices
  remain outside Atlas/status JSON; an atomic measurement-bound sidecar stores
  them under `<ooe>/operator-transport`. Reuse requires the exact measurement,
  capture spec, attention spec, ModelPin, and Atlas ancestry. Missing sidecars
  rerun the probe, matching sidecars save it, and a crash after Atlas append is
  repaired on retry. Run/status expose orphan state and derive `audit_clean`
  from the bank audit.
- Added the exact Qwen MLP evidence bank. BF16 tensors retain their original
  bit patterns in content-addressed binary objects; receipts bind the ordered
  `mlp.input → gate → up → output` stages, verifier evidence, append-only
  history, CAS head, prepared intent, and commit marker. The canonical O1 plan
  is `25/10/5` train/calibration/holdout groups. Fixture and `live-exact` modes
  have distinct resume identities, and fixture data can never enter a
  production `SubspaceCorpus`.
- Corrected the suffix-anchor regression control exposed by Linux: exact state
  parity is now measured against a cold prefix-then-suffix execution with the
  same numerical boundary. One-shot prompt execution remains a separate token
  control because GEMM/GEMV shape changes can differ by float32 ULPs. The test
  still requires bit-exact layer state and snapshot parity on the valid
  iso-boundary comparison.
- Completed the first official local-Qwen operator sidecar wave: five prompts,
  four native heads, 32×32 pre-blend Prefix-Sinkhorn matrices, Atlas revision
  `50`, five capture receipts, inventory `b3ca4b6e…`, and a clean zero-orphan
  audit. Eleven of twelve directed head pairs fail the training condition gate.
  The sole admissible pair `20→8` is selected without holdout access and reaches
  held-out residual `0.0248507190` versus Identity `0.2373913654` and Random
  `0.2729029466` (`9.55×` and `10.98×` improvements). Fit SHA `5c13d631…`;
  holdout SHA `e1f2362a…`; report body `3cd2584b…`; report file
  `6f8cde35…`.

### Exact Markov blankets

- Added a categorical conditional-independence blanket with exhaustive bounded
  power-set search, exact `Fraction` probabilities and Brier scores,
  equal-weight episode metrics, strict chronological train/calibration/holdout
  isolation, singleton/pair synergy checks, Full/Selected/Random/Marginal arms,
  and fail-closed capacity exhaustion. XOR and higher-order interaction controls
  prevent marginal or truncated searches from authorizing a runtime model.
- Added a verified Demand lag blanket. Successful explicit episodes are rebuilt
  in selection/execution order, split only at episode boundaries, and mapped to
  non-contiguous lag atoms. The exact XOR world selects lags `(1, 3)`. Runtime
  priority is validated Blanket singleton, then PPM, then the complete UCB arm
  inventory; residual completion and the external output verifier remain
  mandatory. The receipt-complete E2E benchmark scores Selected/Full `9/10`,
  contiguous PPM `4/5`, Random `1/2`, and Marginal `2/5`, then releases
  `144/336` historical work units through four independently replayed runtime
  paths. Report file SHA:
  `021d54d70c59500bca42e65a20fb81d27241f36c9814eb251cf9176e227d6038`.
- Added exact feature-bound router blankets. A full centroid scan seals the
  input site, global nearest and runner-up before a sparse replay is permitted.
  On all 16 live FERTIG features, the closure is `2/8`: centroid-distance work
  falls from `128` to `32` (`75%`) with bit-exact RouteDecision parity. Live
  report body SHA: `d27cc87e943cbf08512f9863d95964e1e5f8702557145f9659ee9c0e3cac18dd`;
  file SHA: `ee5c7fadc1656522e701c581901bc3339aee93e77a59bdf6b262b40dc527a4b1`.
- Added the Qwen boundary blanket over 200 authenticated transition arrays.
  The selected model is the fixed residual identity
  `layer_output = attention_residual + mlp_output`, reducing six boundary
  candidates to two. Calibration NRMSE is `0.0016464655`; on unseen layer L54
  it is `0.0016575120` versus `0.0017912050` for Full-6 and ranks `1/16` among
  equal-size candidates. All `5/5` leave-one-question-out folds select the same
  fixed `2/6` blanket. Live report file SHA:
  `ed30426134d0d412cac74c3d48fa35fe74c476d2e3ae51a033a41049fed990eb`.
- Added receipt-bound operator transport for IMMER's causal Prefix-Sinkhorn
  matrices. It fits `C A = B C` with per-head/global/identity/random and
  additive-residual arms, rejects unidentified or ill-conditioned maps, and
  replays train/calibration/strict holdout evidence on restore. The deterministic
  mechanism fixture reaches held-out residual `1.33e-17` per-head versus
  `9.57e-4` global. Existing Qwen artifacts fail closed with an exact request
  for the pre-native-blend routed per-head matrix.
- Added a joint Qwen `gate_proj/up_proj` collision-limit evaluator using only
  contextual RMSNormed post-attention evidence and train-only bases/scales.
  In the mechanism control, `k=1` produces `12/4/8` wrong collisions across
  train/calibration/holdout and is blocked; `k=2` is exact on calibration
  `4/4` and holdout `8/8` at `112` bytes. The real Harvester adapter emits a
  sealed instrumentation request instead of treating 40 projected MLP sketches
  as exact gate/up evidence.

### Real FERTIG action learning and controller frontier

- Integration base before the new cut:
  `45778a4aeac55a8f1a7bcb47df73fee4972e1228`.
- Added authority-bound execution learning from real Atlas measurements. Exact
  FERTIG certificates are independently replayed before an `ActionExecution`
  can become a verified teacher transition or change persistent controller
  state.
- Completed `16/16` full warm execution-learning transactions across eight
  promoted O1 sites and two certified input identities. The certified answers
  are `81` and `310`; total Qwen forwards are `0`; all 16 measurements,
  features, executions, learning receipts, and append-only traces are unique.
- Kept the completed source controller immutable at snapshot
  `e59277f837fc5eb8289d24fe431b650d261e24517787cdabc0b96cbc88b81ab1`.
  The learned controller snapshot is
  `a8c50b2979049803e298b4ea600ebbdc698eb2a3c80b6b210264ba6fc9cdd7d4`.
- The private sealed action report has body SHA
  `b6b9b3870a1ac8c79463906e0d421992b9ca0a3b3cbce11ae79d22b8bbb6e598`
  and file SHA
  `d536b54950dfb1027c610953827d0d210f3212d6a40f84897f6e37055f3466e6`.
- Promoted all eight evidence-enriched site policies and exported their exact
  5x5 kernels to authenticated `ComputeCrystal.markov` artifacts. The promoted
  controller snapshot is
  `1562094921890a764c412326336c950046ddd9bf73eaaa9d1f23ca245d8a0fbe`;
  export receipt
  `1663d7524460b9677cde7a5bc92218e2c2429d489964c8bd1a444763fbee590d`;
  frontier
  `44074166d5b7db8b39b43100b95aeffa48e4c80bd114e13f8079150e77315e41`;
  ComputeBank anchor
  `1fa4d2caf5626b33e16e8d5be5d7fc3d7d2d88e34626567d6daacf8f56ce58d8`.
- Every restored exported site policy chooses `execute_fertig` from its
  `execute_fertig` row. The frontier report body SHA is
  `550d707d5093ab3c464bac4d94c068f978430c92def59364b20aca3eb4903715`
  and its file SHA is
  `194f6353067b941a72f6fc36bb1169594856bc5ec69b11950a60d032afdba523`.

### Predictive quotient and formula-derived battery levers

- Added a learned predictive-state quotient using the exact bisimulation
  condition `P(Y_future | h) = P(Y_future | q(h))`. The positive control
  merges four histories into two states; the non-bisimilar control remains at
  four.
- Preserved an exact rational `2/3`–`1/3` future law, reached held-out total
  variation `0`, separated a disturbed law at total variation `1`, and produced
  byte-identical state after reversed observation order.
- Applied the quotient to all 16 live FERTIG learning receipts with the actual
  controller-agent successor `(same site, executed action)`. Nine observed
  `(site, source-action)` states contract to one exact predictive class: `9→1`,
  `9x` compression, one refinement round, zero terminal observations, and zero
  censored observations. Quotient SHA:
  `064dd640c673a3e86dca24513663a1c3a7d2793ed279d2b0b9d2667081e15ae1`;
  persisted file SHA:
  `44e3853cc9bb1953feb01146f4b634a75247cffdc5e129b80e6d7df7c5eaa3fa`.
- The global trace-topology control on the same 16 receipts remains `9→9`.
  The live compression is therefore specific to correct site-local Markov
  semantics, not generic collapsing.
- Audited the external formula archive at SHA
  `5b986886988fe1a1c42256bdebaa6aff9b91588041f2af010ef80408a0b4e0b7`.
  Its normal suite passes `214` tests in `148.20 s` with `560` warnings; a
  warning-fatal file audit passes `24/42` files and fails `18/42`.
- Retained five formulas for native implementation: predictive bisimulation,
  conditional-independence Markov blankets, exact operator conjugation across
  pinned bases, a joint quantized Qwen `gate_proj/up_proj` subspace battery,
  and topology-aware Warmth as a downward charging brake.
- Imported no archive runtime code. Its fixed-`q,K` Softmax Attention path is
  incompatible with IMMER's own causal Prefix-Sinkhorn Attention and stays out
  of the runtime.
- Exact controller forks now copy the complete historical object inventory,
  original manifest bytes, and selected state envelopes under a pinned,
  crash-resumable no-replace intent. Physical aliases, nested stores, foreign
  targets, incomplete forks, stale source pins, staging residue, unrelated
  promotion descendants, extra manifest inventory, forged kernels, and stale
  learning/controller endpoints fail before publication.
- Controller restore rederives every active promoted kernel from verified
  history, consensus, coverage, and router calibration before accepting its
  payload or exporting it.
- Complete warning-fatal regression suite: `1,814/1,814`, `OK`, `591.551 s`.

### First live O1 language frontier

- Completed the official local-Qwen frontier: `40/40` layer probes produced
  `200` authenticated whole-layer/Attention/MLP observations and eight
  persistent PS-Lifted controller Crystal promotions with zero runtime errors.
- Added the exact controller-Crystal export bridge. It replays the persisted
  controller snapshot, Atlas revision membership, model/weight pins, source
  CrystalStore manifest, coverage, calibration, verifier/evidence sets, and
  quantized kernels before publishing eight `ComputeCrystal.markov` artifacts.
- The export receipt embeds the source snapshot and manifest, binds every
  sequential ComputeBank publication, persists its frontier, and restores only
  while source extensions and the append-only bank anchor remain valid.
- Added controller-Crystal language outcomes. The receiver-selected opaquely
  named action and the intended action execute the real stored operators on the
  same float64 state vector. Positive reward requires exact action, artifact,
  and output parity; wrong-site execution is verified negative evidence.
- A mounted executor consumes outcome receipts once, accepts later append-only
  bank growth, rejects forged ancestors, fake controller authorities, replaced
  banks, replay, output tamper, and stale export provenance before any Q update.
- Live result: eight promoted Qwen/O1 site policies became an eight-action
  `ActionFrontier`; a fresh Markov agent reached `100%` frozen vocabulary
  accuracy and then executed `1,000/1,000` unseen greedy operator choices.
- The first live four-action word program has three independently sealed
  support executions. Its recursive lexicon definition compiles to one charged
  Crystal, expands to four real controller actions, releases `3,375` historical
  work units across 25 future states, and matches flat execution within
  `1.110223e-16`.
- Live report body SHA-256:
  `ce5515d29f3fc79e29f1fd37cfae6ea31626ea4e66b0fa616446d1104ff3005e`.
- The reusable Atlas-authenticated CLI independently converges after `2,023`
  episodes, proves three stable greedy cycles, keeps its frozen learner unchanged
  across `12/12` held-out support executions, rematerializes on equivalent bank
  roots, and resumes byte-identically. Report body SHA-256:
  `a2311ccfd0c8a9caf2e74553b6cab3b26b2e75fbfd3f242f6294d868a0f95e8f`.
- Complete warning-fatal regression suite: `1,793/1,793`, `OK`, `617.930 s`.
- Added the typed controller-support portability bridge. Three isolated
  four-step occurrence chains now authorize one generic macro-discovery receipt
  only when their frozen snapshot, global words, actions, definition,
  controller export, numerical replay, and verifier authorities all agree.
- A second independently initialized live agent converges on the same eight
  O1 actions with `8/8` different word assignments. Semantic translation remains
  `8/8`; the first live macro localizes into four sibling-native words, compiles
  locally to one charged Crystal, releases `3,375` historical work units, and
  matches flat execution within `2^-52`. Live transfer report SHA-256:
  `7b4a026da4091b7cae010c354512a9a7fc2e7b80aa86dadda8c50e9ecb4a5334`.
- Complete warning-fatal suite with enforced controller authorization:
  `1,794/1,794`, `OK`, `559.730 s`.

### Portable executable dialect mesh

- Added pairwise dialect alignment over consequence-grounded action semantics.
  Agents keep independent opaque words; translation matches only actions whose
  complete `ActionBinding` is byte-identical in source and target frontiers.
- Translation receipts bind both snapshot/frontier pairs, both explicit
  contexts, every source/target word pair, its executable action, and all
  unmapped actions. Execution recomputes the alignment from the bound snapshots
  before translating, so a rehashed handcrafted word permutation is rejected.
- Added portable semantic word programs. A verified discovery is converted from
  source words into an ordered action program plus exact action bindings,
  discovery/definition hashes, authorities, and supporting trajectory hashes.
- A target dialect localizes the semantic program through its own learned
  words, verifies every target ActionBinding against the source semantics,
  creates a context-bound target-native `ExecutableWordDefinition`, and
  compiles against context-local primitives in the target ComputeCrystal bank.
- Target authorities must equal the portable program's verifier authorities.
  Changed, missing, context-unexpressible, non-identical, or authority-rebound
  target actions fail before definition installation. Context identity is part
  of definition/state hashes and every append-only lexicon transition.
- The benchmark uses a fully shared six-token inventory, so direct transfer is
  not an out-of-vocabulary trick. Every context is learned by an independent
  `ConsequenceMarkovLanguage` and merged only through exact snapshot hashes.
- Five-seed mesh result: `5/5` surface dialects are unique; all four target
  dialects have three distinct context mappings and zero globally stable action
  words. Verified translation scores `60,000/60,000` unseen programs, including
  `100%` in the worst target context. Direct surface transfer scores `6.84%`
  programs / `23.41%` tokens; a non-identity permutation placebo scores
  `1.42%` / `19.58%`.
- One portable macro localizes, compiles, and executes exactly in `75/75`
  dialect-context targets. Altered bindings are rejected on `5/5` seeds and
  discharge releases mean `150` historical work units across 25 future inputs.
- Complete warning-fatal regression suite: `1,778/1,778`, `OK`, `905.252 s`.

### Production language bridge and continual self-extension

- Added exact action-frontier builders for materialized ComputeOperatorGraph
  routes and active algebra candidates. Route actions bind immutable route
  receipts; algebra actions bind exact programs, candidate schemas, router
  state, archive state, verifier, and a hashed reward policy.
- Added `ConsequenceLanguageBridge`. A receiver-selected route now executes
  through the real `DemandRoutedExecutor`; only the fully joined and persistently
  committed `DemandRoutedExecutionReceipt` can create language feedback.
  Rejection remains genuine negative evidence, while operational abort creates
  no quality update.
- Added atomic production-language persistence with CAS, rollback anchors,
  transaction rollback after publication failure, exact bound-abort state, and
  self-describing frontier restore. Macro-promotion metadata publishes before
  the promoted state head and is recoverable by the promoted state SHA.
- Added strict frontier migration. Q evidence transfers only for byte-identical
  action bindings under the same reward policy and authenticated authority
  transitions. Context-table evidence transfers only under the same context
  schema. Changed actions, reward policies, or unproved authorities reset.
- Added append-only route-frontier proofs carrying every canonical graph-state
  payload. Restore replays every graph append, checks endpoint authorities,
  retained/added route inventories, and the exact set of changed authority
  names. Hand-built name-only proof objects cannot retain evidence.
- Removed the vocabulary capacity cliff. Frontier migration preserves every
  existing word and deterministically allocates unused opaque words until the
  vocabulary covers the expanded action set; copied Q tables use exact old-word
  slices and new words start unvisited.
- Added verified word trajectories ordered by persisted Demand selection time.
  Promotion requires the declared step count, all-positive joined outcomes,
  one episode identity, and a terminal verification receipt whose verifier is
  pinned by the frontier.
- Added repeated-word discovery, route-word-to-program resolution, MacroOption
  translation, local lexicon compilation, compiled-word action promotion, and
  atomic language-frontier migration. The promotion receipt binds discovery,
  definition, compiled artifact, parent/next frontier, migration, and persisted
  state.
- Added generalized snapshot compute resolution for direct Crystals, programs,
  and materialized routes. A foreign graph state is rejected even when the
  current snapshot contains only direct compute artifacts.
- Five-seed continual-growth result: base actions `100%`; old-action retention
  after promotion `100%`; new action initially abstains on every seed, reaches
  learned execution after `328` episodes on average, and final four-action
  accuracy returns to `100%`; a fresh no-memory model scores `0%`.
- Context-schema shift discards local tables while global meaning remains
  `100%`. Reward-policy shift resets `4/4` actions and scores `0%`. A
  second-generation word compiles to one Crystal on every seed and increases
  historical work released across 25 future states from `400` to `1,200` units.
- Complete warning-fatal regression suite: `1,774/1,774`, `OK`, `733.121 s`.

### Consequence-grounded language and recursive compute words

- Added a context-aware Markov signalling runtime. The receiver API accepts
  only an opaque word, a context ID, and exploration; one-shot verifier
  feedback contains the chosen-action receipt and scalar consequence but no
  sender intent, target action, target state, or semantic label.
- Combined a shared word/action table with context residual tables. Stable
  meaning transfers into unseen contexts while the same word can learn
  different actions in different authenticated contexts. Valid in-vocabulary
  words abstain when unvisited, low-value, or ambiguous.
- Added immutable action frontiers for Crystal, program, route, option, organ,
  causal-site, residual-plan, and Qwen-path actions. Each frontier binds its
  actual action/context schemas and named authorities without imposing a
  universal model-specific checklist.
- Added canonical learner state, deterministic RNG resume, frozen language
  snapshots, one-shot pending-decision replay rejection, fixed-teacher cultural
  induction, and arbitrary multi-slot factorized grammar trained from one
  whole-action consequence.
- Added the self-hosting executable lexicon. Definitions bind snapshot,
  frontier, and authority hashes; primitives are immutable; unknown
  dependencies, cycles, conflicts, stale definitions, tamper, and namespace
  collisions fail closed. Recursive repair installs each missing definition
  once.
- Added bottom-up word-DAG compilation into the real `ComputeCrystalBank`.
  Homogeneous children use duplicate-preserving exact fusion; mixed families
  retain a bounded ordered `ComputeProgram`. A word claims constant discharge
  only when the compiler produces one charged Crystal.
- Added opt-in transitive fusion-work provenance. Legacy Crystal bytes remain
  unchanged; recursively fused words carry the complete authenticated source
  work into `ComputeChargeReceipt` and VM discharge accounting. Forged
  rehashed work metadata fails parent-chain reconstruction. Legacy fused
  parents remain executable but cannot mint a new transitive charge until their
  primitive lineage is recompiled with provenance.
- Added an append-only lexicon bank with immutable history objects, commit
  markers, CAS head, crash recovery, concurrent-writer exclusion, rollback and
  fork detection, and optional external trusted-head anchoring.
- Fresh five-seed runtime benchmark: primitive language, 1,000 unseen programs,
  context-dependent meaning, held-out grammar, option words, cultural transfer,
  and self-hosted macros all score `100%`; shuffled semantics score `2.02%`,
  a fixed no-message policy scores `0.98%`, no-action abstention `2.90%`,
  shuffled grammar `5.00%`, and the holistic held-out table `0%`. Every
  unassigned in-vocabulary word abstains.
- The depth-12 word DAG stores `36` references for `8,191` primitive actions,
  compiles to one constant-discharge Crystal, matches the flat program exactly,
  and measures `8.380x` faster under the same authenticated VM boundary. Across
  25 future states it releases `1,638,000` historical work units and performs
  `200` live work units.
- Complete warning-fatal regression suite: `1,763/1,763`, `OK`, `628.021 s`.

### Causal sublayer and prompt-preserving sequence intelligence

- Added an isolated Qwen layer-boundary observer. Cartography now records five
  projected boundaries per measured layer: normalized Attention input,
  Attention output, Attention residual, normalized MLP input, and down-projected
  MLP output. The observer receives clones, cannot mutate model math, filters
  stages before allocation, and changes neither checkpoint reads nor committed
  continuation state.
- Converted those boundaries into four prompt-diverse operator families:
  Attention core, Attention residual, MLP core, and MLP residual. The existing
  whole-layer transition remains a fifth family; every array is hash-bound to
  the sealed Qwen evidence document.
- Added `CAUSAL_MIX_FLOAT64`, a strict lower-triangular row-normalized sequence
  operator with ABI `[..., T]` and left action `...k,qk->...q`. Homogeneous
  kernels fuse as `K_following @ K_current`, retain exact zero future mass, use
  a dedicated fusion verifier, and enter graph charging without inheriting the
  semantically different Markov contraction ledger.
- Added the prompt-preserving contextual sequence instrument. It predicts
  `Y = X + ΨC` from RMS-normalized input, a fixed seeded multi-timescale
  recurrence, and optional quadratic reservoir features. Recurrence resets per
  prompt; training prompts fit the readout, one later prompt selects the feature
  map/ridge, and a separate final prompt produces the holdout receipt.
- Bound raw X/Y content independently of prompt/evidence labels, rejected
  renamed duplicate feature maps, and added a deterministic aggregate memory
  preflight before substrate, feature, or Gram allocation. Fits and holdout
  receipts enforce `predictive_only=True` and never enter the executable
  ComputeOperatorGraph.
- Fixed synthetic holdout: selected sequence MSE `6.726089e-05` versus tuned
  pointwise `5.803816e-02` (`862.881x` lower); shuffled-token and shuffled-output
  placebos score `3.1959` and `3.1541`.

### Living O1 frontier and executable harvested algebras

- Replaced one-shot cartography closure with additive frontier growth. The
  immutable initial Qwen manifest now accepts hash-chained prompt/job events;
  `O1Cartographer.extend_jobs()` atomically reconciles those cells into the
  existing scheduler while retaining every outcome, feature history, replay
  item, stream snapshot, and Atlas record.
- Added deterministic Qwen layer/site grids and full cross-product expansion.
  Mixed prompt-plus-job appends materialize both `new prompt × old jobs` and
  `all prompts × new jobs`; inherited job-specific family and semantic evidence
  bindings remain exact.
- Bound attention coordinates to the authenticated Qwen hybrid topology.
  `attention-q` now resolves `linear_attn.in_proj_qkv` on DeltaNet layers and
  `self_attn.q_proj` on full-attention layers from the local bundle config;
  the first live L45 failure exposed and closed the stale all-softmax mapping.
- Made contextual evidence prompt-diverse across the weight map. Multiple
  coordinates from the same prompt still enter Atlas, but only the first
  transition counts inside one operator family; later sites reject as
  `duplicate-prompt-in-group`. Fit and temporal holdout therefore require three
  distinct prompt signatures instead of three coordinate receipts carrying the
  same hidden arrays.
- Added the durable `idle` runner and weight-free `frontier-status`. Idle cycles
  resume after process failure, consume later frontier events, persist exact
  run/wait/error accounting, and reject a journal rollback once the scheduler
  has anchored the added cells.
- Generalized `OperatorAlgebraCandidate` from exact affine monoids to the strict
  union `AffineProgram | ComputeProgram`. New v2 candidates bind their runtime,
  program bytes, endpoint ABI contract, discovery verifier, execution verifier,
  and program receipt; legacy v1 affine candidates restore byte- and
  hash-identically.
- Added the authenticated harvest-to-algebra bridge. A held-out affine,
  permutation, or Markov promotion is rebound to its exact graph edge and bank
  artifact, published as a `ComputeProgram`, admitted into the contextual
  Thompson/MAP-Elites catalog, executed by `ComputeCrystalVM`, externally
  verified, and returned to the router as exact positive or negative feedback.
- Persisted the bridge itself. Profiles, graph/edge/program bridge receipts,
  and candidate admission records are content-addressed before router CAS;
  restart audit reconstructs every active routed candidate. A fault-injected
  router-CAS failure leaves reusable immutable artifacts and completes exactly
  on retry.
- Derived execution-verifier identity per harvested family from discovery
  group, source/target state, operator kind, and input/output ABI. Discovery
  fitting and runtime result judgment remain separate contracts.
- Added explicit-path planning and discharge. `plan_exact_path()` preserves the
  selected edge sequence across parallel same-endpoint alternatives, and
  `discharge_exact(route_sha256, value)` executes that authenticated route
  instead of reselecting the canonical cheapest path. One-edge programs never
  claim charge savings; multi-edge charged routes retain exact work accounting.
- Added the fixed harvested-program intelligence trial. The contextual agent
  selects three executable affine/permutation/Markov families correctly on
  `174/180` decisions and `60/60` late decisions; the information-destroying
  shuffled-context placebo reaches `49/180` and `13/60` late.
- Verified the complete repository through the prompt-diversity ingestion cut
  with `1,709/1,709` tests under fatal `ResourceWarning` in `574.828 s`. The
  reviewer-added fail-closed legacy-state invariant raises the discovered total
  to `1,710`; the complete affected OoE block passes `368/368` in `162.255 s`.
  Ruff, formatting, compile, JSON, Markdown links, private-path, diff, and
  anti-hedge gates pass.

### Demand-routed execution and exact operator algebras

- Added `DemandRoutedExecutor`, the production feedback loop over materialized
  compute. One canonical receipt joins PPM prediction, persisted UCB1 choice,
  the exactly forced charged prefix, live residual suffix, both VM receipts,
  external quality verification, computed savings reward, and the committed
  positive or negative demand outcome.
- Made operational failure neutral. VM exceptions, verifier crashes, stale PPM
  state, and pre-outcome commit failures append a verifier-bound
  `SelectionAbortEvent`; the event removes the unfinished pull without
  manufacturing negative quality evidence. Aborted selections cannot later
  accept an outcome.
- Bound the joined state chain exactly. PPM state equals the selection pre-state;
  selection post-state equals outcome-transition pre-state; the transition
  generation and deterministic `OutcomeEvent` hash must match the embedded
  outcome. Same-length fork and receipt-splice attacks reject.
- Added the exact guarded affine-monoid runtime. Every action is an integer or
  per-coordinate modular pair `(A,b)`; composition is
  `(A₂A₁, A₂b₁+b₂)`. Guard-stage traces preserve partial domains through
  fusion, and explicit phase, depth, dead, bit, and logical-capacity fields
  replace the old learned/float approximations.
- Added an explicit LIFO stack, a genuine two-counter `aⁿbⁿcⁿ` DFA, dual
  modular rolling fingerprints with length/separator/phase binding, strict
  signed Decimal-Horner through 128 digits, and additive/multiplicative/cyclic
  group-map bridges. Multiplicative zero, stack overflow/underflow, malformed
  grammar, out-of-order equal-count strings, and bit-bound overflow fail closed.
- Added the mandatory exact fingerprint boundary. Modular equality creates a
  candidate only; a byte-verifier receipt decides exact equality. The fixed
  collision trial reaches the candidate state and is correctly rejected by the
  exact verifier.
- Added one replay-verified `ExecutionBundle` and `AffineMonoidBank`. Program,
  initial/final state, state/program/execution receipts, and verifier lineage
  persist as a single atomic content-addressed payload with crash retry,
  idempotence, tamper detection, and exact replay on restore.
- Added the contextual algebra meta-agent. Verified Stack, Fingerprint, and
  Decimal candidates live in a Pareto MAP-Elites catalog; contextual Thompson
  updates learn the correct algebra per task and recover after regime change.
  Handcrafted selections cannot poison posteriors because every update
  reconstructs the exact parent router and deterministic Thompson choice.
- Added receipt-joined heterogeneous ensembles. Programs with different state
  schemas execute as independent parallel lanes; no sequential/common ABI is
  claimed. Each lane embeds its exact program and initial state, replays
  standalone, and rejects cross-program, same-schema, swapped, missing, or
  verifier-artifact splices.
- Fixed-seed results: demand rewards `1.25` and `1.50`, with the deeper prefix
  selected on decision three; stack depth `53/64` exact and depth `65` dead;
  `aⁿbⁿcⁿ` unseen counts `4/4`, placebos `7/7`; contextual algebra routing
  `120/120`, shuffled-context placebo `0/90`, Stack→Decimal recovery `30/30`;
  three heterogeneous lanes replay exactly under one joined receipt.
- Verified the complete repository with `1,673/1,673` tests under
  `ResourceWarning`-as-error in `549.699 s`. All changed files pass Ruff,
  compile, diff, anti-hedge, link-target, JSON, and private-path gates.

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
  cache. Qwen remains the primary local full-model substrate and novelty path.
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
