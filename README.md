# IMMER

**Research trial artifact for causalized local frontier inference.**

IMMER studies a single question: how far can a local system push frontier-model
weights when neural inference, deterministic verification, persistent state,
and causal weight addressing are designed as one runtime?

The repository accompanies the research line around `.causal`, deterministic
validation, Causal Prefix Sinkhorn Attention, and O(1)-state systems. It contains trial code,
tests, and selected public receipts. Model weights, learned graphs, private
traces, capability-transfer material, and deployment state remain outside the
release surface.

## Research focus

- **Causalized local weights.** Immutable tensor payloads are paired with an
  appendable causal address graph inside one local model bundle.
- **Local target execution.** Qwen3.8-27B runs directly from causalized local
  weights with stateful full-attention and DeltaNet continuation. Ordinary
  decode replaces continuation state layer by layer, while stable local
  ranges fill Torch-owned weight storage directly through `preadv`.
- **Native packed execution.** A source-bound derived weight plane maps the
  original causal graph onto row-addressable Q4_0/Q8_0 files without changing
  the BF16 bundle. Every text projection executes as Q4; embeddings and the LM
  head use Q8.
  The C kernel consumes mmap pages directly with AVX2 and retains the
  existing Qwen Attention, DeltaNet, Sinkhorn, state, tokenizer, and FERTIG
  paths. The complete text plane is 16.40 GB instead of 53.79 GB. Consumed
  read-only pages leave process RSS at every existing layer boundary; the Q8
  LM head performs one native bounded Top-K scan and releases each completed
  interval from inside the kernel. On the 16-core AVX2 deployment, arbitrary
  Full-Q4 chat now peaks at 1.17 GB instead of 16.01 GB while preserving the
  exact generated token trace. F16C scale conversion brings the two-token
  reference generation to 8.66 seconds without changing its trace or RSS.
  Gate, Up, BF16 SwiGLU, activation quantization, and Down now execute inside
  one native OpenMP team. A complete 65,536-entry BF16-SiLU table is generated
  from the installed Torch semantics, keeping the fused path bit-identical
  across host libm implementations. AVX2 hosts with F16C decode every packed
  FP16 block scale in hardware; other hosts retain the scalar IEEE reference.
  Speculative Q/K/V and DeltaNet input projections also share one Q8 input and
  native team across all staged token rows; K2 removes 176 redundant native
  calls and 352 row quantizations per target wave.
- **Direct Q4 MLP-page execution (explicit).** An opt-in persistent controller
  learns 64-neuron activation-page routes from exact full-Q4 MLP traces. Its
  temporal, cross-layer, and marginal agents use Fixed Share to select a
  bounded page set; unready layers execute the unchanged full MLP, and staged
  continuation updates commit or roll back with the accepted target prefix.
  Enable it explicitly with `--mlp-page-state` and bound it with
  `--mlp-page-width`; the default chat command never enables this route.
- **Predictive weight transport.** A persistent two-agent operation Markov
  model learns the exact tensor/range sequence already emitted by the
  Streamer. Order-1/2 Fixed Share, surprise regimes, and Ricci retention
  roll a bounded multi-step beam over future grouped operations. Path
  probability, Ricci value, and reuse distance rank a deduplicated leaf
  reservoir under one global byte/leaf budget. Accepted hints become delayed
  actions: later exact demand supplies useful/hinted bytes to persistent
  distance agents, which reweight the beam and scale its budget between 12.5%
  and 100% of the hard maximum. Linux warms it via
  `POSIX_FADV_WILLNEED`: no payload copy, logical read, budget charge, or pager
  weight cache is introduced.
- **Native local drafting.** Deployed chat uses the zero-weight Markov council;
  causalized Qwen3.5-0.8B and embedded MTP providers remain explicit modes.
  The provider proposes transactional continuations in a configurable K=2–16
  target window, and Qwen3.8 alone verifies and commits them. The Council
  combines live online distributions with retained Atlas branches in a
  bounded beam, so a stronger multi-token path can win without treating a
  ranking score as acceptance probability. Target-confirmed episodes become
  reusable variable options up to
  15 tokens;
  global and contextual-dialect agents compete by support, confidence, context
  depth, and dialect similarity. Under an adaptive request ceiling, the
  Council emits one maximum tail with prefix-local confidence/disagreement;
  every target wave selects direct K1 or K4/K8/K16 and stages only that prefix.
  Confirmed prompt/output boundaries additionally induce bounded token
  programs made from literal spans and relative prompt-copy actions. Programs
  with different internal bindings execute only when they agree on one
  concrete continuation for the new prompt.
  On the compute-bound packed CPU plane, backend row cost prevents marginal
  confidence from multiplying neural work; K1 learns from the confirmed token
  without cloning transactional state. Existing target receipts can be
  imported atomically and idempotently; normal persistent traffic then grows
  the same episode, dialect, expert-rapidity, and phrase memory online.
  Within one request, an ephemeral high-order agent also learns repeated answer
  contexts from confirmed output and corrections. It never trains on prompt
  tokens or rejected drafts. A sibling phrase/copy agent promotes the common
  continuation of repeatedly confirmed spans into a verified multi-token
  option. Both agents use a bounded 4,096-token horizon, and durable request
  state is written once at successful finalization. The eight-expert Council
  also updates its combined Rapidity/Fixed-Share weights after every confirmed
  row, so the next decision already uses the experts that are winning inside
  the current answer. A request-local 16×8 Beta overlay simultaneously learns
  which experts win at recursive proposal positions 0–15. Per-position
  lookahead regret also compares every planned token with its greedy baseline,
  allowing a harmful override to shut off before the answer ends.
- **Embedded MTP drafting.** The target checkpoint's own one-layer Qwen3.5 MTP
  branch now runs through the same causal Q4 bank, shared embedding, native
  bounded LM head, and exact rolling target verifier. Shifted token embeddings
  pair with the preceding final-normalized target hidden state; accepted draft
  prefixes alone advance the private MTP attention state. A persistent
  first-order Markov/Beta calibrator learns reliability by proposal position,
  logit-gap bucket, and previous outcome. The cost-aware selector chooses
  K1/K2/K4/K8/K16 from verified prefix yield, so raw MTP confidence cannot open
  an uneconomic target wave. Adaptive execution computes only the proposal
  frontier selected by that policy. On an arbitrary 16-token counting request,
  K2 accepted eight drafts and reduced exact generation from 34.05 to 26.74
  seconds while cutting complete target forwards from 16 to 9. In explicit
  hybrid mode, chat runs a round-wise Markov/MTP Council. Markov gets first
  refusal on every response round; MTP covers novelty, and the next round can return to
  Markov as soon as a learned answer continuation applies. Both providers
  consume every exact target-confirmed prefix, so switching requires neither
  target replay nor another prefill. When MTP owns a novelty round, the unused
  Markov proposal remains a shadow prediction: target verification scores its
  causally valid prefix through the first mismatch and updates every Council
  expert. Fixed-Share weights then convert each agreeing expert's persistent
  Beta accuracy, evidence maturity, and current Top-1/Top-2 margin into draft
  confidence; unknown-dominant, near-tied, cold, and forced-phrase predictions
  receive no empirical inflation. State v7 separates those posteriors for all
  eight experts at draft positions 0–15, with global experience used only
  until a position acquires its own target-confirmed evidence. After the first
  shadow mismatch, the zero-weight Council re-predicts later verified tokens on
  their actual prefix, so deeper positions learn without scoring a
  counterfactual Markov suffix or running Qwen again. It also predicts a new
  carry from the complete accepted MTP prefix; the next target token therefore
  trains the next recursive position even when Markov's original first token
  was wrong. If Markov is not strong enough to own a round but shares an exact
  computed prefix with MTP, Hybrid v9 can contribute only its positive,
  disagreement-discounted confidence surplus. Fusion stops at the first token
  divergence and cannot revive MTP's zero-confidence padding. Each recursive
  position also blends global Rapidity with its own Beta-normalized specialist
  weights according to evidence maturity, allowing different Markov experts to
  own token selection at positions 0 and 1 while retaining Fixed Share. State
  v8 carries the same 16×8 matrices inside every prompt dialect, so only
  evidence from the current similar dialect at the current recursive position
  can specialize the mixture. Inference uses a four-profile dialect Council:
  the active learning dialect keeps one seat, while the remaining seats are
  ranked by similarity × visits × Ricci age; prompts below the activation
  threshold stay global. A bounded one-step planner evaluates the current top
  four tokens against their position-/dialect-specific next state and overrides
  greedy choice only after a positive discounted log-probability margin.
  Novel MTP requests teach their complete target-confirmed episode back to the
  Council. After two confirmed copies, the same context can execute from its
  exact dialect phrase: a live third request used no MTP, accepted seven Markov
  drafts, and completed 16 output tokens in nine target forwards. The persistent
  Council corpus contains generated answers rather than chat templates or user
  prompts. Embedded MTP evaluates its complete high-confidence proposal
  horizon, and existing v1 calibration state migrates into the current provider
  without discarding learned reliability. On Beast, an unseen German
  metal-spoon request produced 24 tokens in 16 target forwards, saving eight
  forwards (33.3%). A second unseen one-sentence request produced `Eis schwimmt
  auf Wasser, weil es eine geringere Dichte als flüssiges Wasser besitzt.` in 22
  tokens and 11 target forwards, with 11 accepted drafts; its K4 round accepted
  all three staged drafts.
- **All-layer Fast MLP.** A prompt-free weight initializer derives p4/k32
  pilots for every Qwen MLP layer from exact Gate/Up Gaussian joint moments.
  Existing full-path tensors update bounded recursive-ridge route statistics;
  the resident exact Down matrix calibrates a one-row sparse shadow and scalar
  correction without another source read. Capture, output error, transport
  economics, width, warmup, and periodic checks select the unchanged full MLP.
  [All-layer runtime](docs/FAST-MLP-ALL-LAYER.md).
- **Exact causal LM-head rail.** A weight-only residual-PQ tree supplies
  certified upper bounds for canonical token pages. Proven-impossible pages
  require no checkpoint read; surviving pages apply the actual stored PQ code
  tuple and residual radius per vocabulary row, then read only the K<=16 union
  of rows still capable of winning when that removes at least half the page in
  at most four contiguous runs. A call with no physical row saving in its first
  64 leaf probes stops evaluating row caps. The unchanged page-shaped
  FP32-accumulate/BF16-output scorer preserves stable lower-token-ID ties.
- **Grounded composition.** [FERTIG](https://github.com/DT-Foss/FERTIG)
  supplies deterministic parsing, verification, and explicit abstention.
- **Organism of Experts.** O1 measurements become authenticated Atlas evidence;
  PS-Lifted Markov agents learn action-conditioned world models, continuously
  expand the live weight frontier, harvest real contextual transitions, choose
  executable operator algebras, and learn from verified execution outcomes.
- **Receipt-native neural Seed.** A small GRU + Causal Prefix Sinkhorn +
  Keyed-SwiGLU substrate learns contextual actions, novelty, predictive
  quotients, route value, and expected work from authenticated Qwen/O1/Atlas
  receipt sequences. D-optimal acquisition chooses the next measurements; the
  Seed proposes while the existing verifier path remains executable truth.
- **Consequence-grounded executable language.** Separate sender and receiver
  policies ground opaque words from authenticated action consequences alone.
  Shared semantics generalize across contexts, context residuals learn genuine
  state-dependent meaning, factorized word slots recombine unseen actions, and
  valid but unvisited or ambiguous words abstain.
- **Self-hosting compute words.** Known executable words define new words as an
  immutable DAG. Missing definitions repair recursively, homogeneous
  Crystal children compile bottom-up without flattening, and the final word
  discharges one authenticated operator while retaining its complete
  transitive work provenance.
- **Continual language growth.** Real `DemandRoutedExecutionReceipt` outcomes
  update the language atomically. Verified episode sequences become compiled
  words, compiled words become new frontier actions, unchanged actions retain
  their evidence across proven append-only revisions, and new actions receive
  deterministic unused words and begin behind abstention.
- **Live O1 vocabulary.** Promoted PS-Lifted controller kernels export into an
  append-only ComputeBank with their complete Atlas/model/coverage provenance.
  Markov agents ground opaque words by executing those real operators; repeated
  accepted word programs compile into charged Crystals without flattening.
- **Portable dialect mesh.** Independently grounded agents retain their own
  opaque surface conventions while translating through identical executable
  ActionBindings. Verified macro programs move as semantic action sequences,
  localize into each target vocabulary, and compile locally without copying
  sender Q-tables or requiring shared word IDs.
- **Compute batteries.** Already-paid computation is retained at its natural
  granularity: results, continuation states, numerical operators,
  factorizations, routes, prefixes, and residual states. Native Qwen
  continuation restore is one executable instance of this wider stored-compute
  system. An explicit prefix cache now charges its longest shared chat-template
  state on the first miss and can restore it underneath the Markov rolling
  decoder; short prefixes remain direct when restore cost exceeds saved work.
- **Stored compute.** `ComputeCrystal` programs materialize reusable numerical
  operators. The Markov operator graph charges verified routes, fuses compatible
  chains, applies the deepest charged prefix to previously unseen values, and
  computes only the unknown residual suffix live.
- **Autonomous operator search.** Constructive Birkhoff decomposition stores
  only the permutation atoms an operator uses. Contextual Thompson mutation,
  behavioral MAP-Elites, Fiedler-projector novelty, verified UCB1, PPM route
  prediction, co-occurrence prefetch, and Ricci retention drive exploration and
  demand.
- **Exact operator algebras.** One guarded affine-monoid runtime executes stack
  shifts, multi-counter DFAs, rolling fingerprints, Decimal-Horner, and group
  maps over exact integer or modular rings. A contextual Markov meta-agent
  learns which algebra fits each task and joins incompatible programs only as
  independent parallel lanes.
- **Living cartography.** An append-only frontier journal adds prompts, layers,
  weight sites, and interventions without rebuilding the Atlas or losing O1
  replay. A durable idle runner consumes new cells whenever compute is free.
- **Harvest-to-execution routing.** Held-out affine, permutation, and Markov
  discoveries become tagged `ComputeProgram` candidates in the same
  Thompson/MAP-Elites router. Exact edge choice survives materialization and
  discharge; profile, bridge, admission, program, and verifier evidence remain
  restorable after a crash.
- **Sublayer cartography.** The Qwen probe captures hash-bound projected
  Attention input/output/residual and MLP input/down-projection/residual
  boundaries during the existing forward. The observer is filtered before
  cloning and adds no checkpoint reads or model-state mutation.
- **Causal sequence compute.** `CAUSAL_MIX_FLOAT64` stores a realized causal
  sequence kernel as the left action `K·V`, preserves exact zero future mass,
  fuses in execution order, and participates in ordinary route charging under
  its own verifier identity.
- **Prompt-preserving prediction.** A fixed multi-timescale recurrent substrate
  plus quadratic readout predicts Qwen residual sketches without flattening
  prompt boundaries. Train, validation, and final holdout are separate,
  content-duplicate relabeling rejects, and the artifact is explicitly
  predictive-only.
- **Exact result cells.** A `ResultCell` is the sealed special case for one
  complete Qwen/FERTIG result binding. It is distinct from a generic
  `ComputeCrystal` and executes only under exact prompt, model, provenance, and
  final-parity contracts.
- **Transport intelligence.** Label-free route observations support held-out
  Markov and placebo studies while the model's official router remains
  authoritative.
- **Independent mechanisms.** IMMER includes its own Causal Prefix Sinkhorn
  Attention line,
  persistent O(1) state, and digest-bound exact capability organs.

## Local packed runtime

```bash
# On the IMMER server this auto-mounts local Qwen, full Q4/Q8, native
# DeltaNet and FERTIG.
PYTHONPATH=src python3 -m immer chat "<arbitrary text>"

# Keep the mounted model alive for multiple raw-text or JSONL requests.
PYTHONPATH=src python3 -m immer chat --jsonl

# Emit one final machine-readable receipt instead of live text.
PYTHONPATH=src python3 -m immer chat "<arbitrary text>" --output json
```

Outside the canonical deployment, set `IMMER_QWEN38_ROOT` and optionally
`IMMER_QWEN38_Q4` and `IMMER_QWEN38_FAST_MLP`. Explicit CLI paths remain
available as overrides. The sparse MLP plan is explicit because full Q4 keeps
the general-chat language intact. `--raw-qwen` bypasses the normal
FERTIG-first route. Normal single-request chat shows live token progress while
stdout remains exactly the final FERTIG-routed answer. `--raw-qwen` streams the
actual decoder text, and `--no-stream` buffers either mode until completion.

When a verified local OoE warm bank is present, chat mounts it before opening
Qwen. An authenticated ResultCell hit executes with zero Qwen forwards and
persists the Markov accounting; an unknown prompt falls directly through to
full Q4. `--no-ooe-warm` disables this route, and `--ooe-warm-root` mounts an
explicit bank outside the canonical deployment.

The canonical Q4 deployment keeps the existing MTP-capable v3 target bank but
uses its persistent token-level Markov Council directly. It does not load or
run embedded MTP unless `--draft-mode mtp` or `--draft-mode hybrid` is supplied.
Atlas and online agents expand a bounded beam, while a request-local high-order
agent can reuse repeated context learned earlier in the same confirmed answer.
A request-local phrase agent can copy the deterministic common prefix of two or
more prior confirmed continuations directly into the same verified K window.
Full Q4 still verifies every emitted token; `--no-markov-draft` forces direct
K1 decoding. Direct Q4 MLP-page routing is also explicit-only through
`--mlp-page-state` and is disabled in this default.

Markov state v6 also retains prompt/output boundaries for composition. Two or
more distinct target-confirmed bindings can induce literal/copy programs over a
bounded 64-token prompt suffix. An unseen slot is materialized as ordinary
draft tokens and remains subject to the same exact Qwen prefix verification;
ambiguous or conflicting programs abstain rather than execute.

Cold general chat now grows the warm bank automatically. The first successful
Full-Q4/FERTIG path charges an exact ResultCell, derives a prompt-native Markov
feature, teaches `qwen_fallback → mount_organ`, promotes the per-runtime
controller, and persists the index. A repeated request with the same runtime
profile, question and rendered-token identity then executes the stored cell
without loading Qwen. FERTIG mismatches are never charged; profile, token,
model, Q4, tokenizer and current runtime-code changes produce a cold miss.

The same cold stream induces parametric Markov programs when two distinct
ResultCells prove the same deterministic span transformation. Copy and case
operators are promoted from whole-result consequences, not predefined prompt
phrases. A new slot under the learned static context can then execute without
Qwen. Conflicting transforms, insufficient support, FERTIG mismatches, wrong
prompt-token identity, non-ASCII output, or output beyond `max_new_tokens`
abstain. Executable template descriptors live only in the explicitly private
per-profile store and are revalidated against their source ResultCells.

## Selected trial evidence

| Trial | Result | Scope |
|---|---:|---|
| Local causal bundle reopen | 77.77 → 1.20 s; 64.90x | complete 55.6 GB Qwen3.8 bundle; first full-content verification followed by unchanged next-process stat+digest reuse; no model forward |
| Native causal Q4/Q8 chat | readable arbitrary German output; TTFT 11.95 s; 8 tokens in 20.12 s | real 27B CPU run on 16 AVX2 cores; 498 text matrices; 16.02 GB peak RSS; 16.40 GB derived payload; two prior token traces preserved exactly |
| Cost-aware Markov abstention | 74.58 s vs 79.05 s direct; identical token trace | seven K1 waves on an unseen German request; zero draft bytes/linears; confirmed episode still updates Council state |
| Persistent JSONL chat | second request opens 0 tensors | one process retains the verified mmap plane; observed one-token generation 15.01 s then 16.90 s |
| Native DeltaNet K=4 convolution | 79.05 → 23.01 s generation; 3.44x | direct causal channelwise kernel replaces generic 10,240-group Conv1d; identical output/token trace; TTFT 25.05 → 13.65 s |
| Adaptive embedded MTP | 34.05 → 26.74 s generation; 16 → 9 target forwards | arbitrary 16-token counting request; identical target trace; eight accepted drafts; eight provider head scans; 1.58 GB peak RSS |
| Round-wise arbitrary chat | 22 tokens in 11 target forwards; 11 drafts accepted | unseen German one-sentence request; complete correct answer; one K4 wave accepted all three drafts; 33.11 s generation; 1.46 GB peak RSS; deeper MTP positions learned online |
| Shadow Council learning | 14/14 MTP rounds returned to Markov | unseen moon question; 17 target-confirmed Council feedback events, 10 directly scored shadow tokens, 24 output tokens in 15 target forwards; ordinary chat traffic trains Markov without extra Qwen work |
| Reliability-driven Markov takeover | one Markov round inside an unseen response | unseen glass question; learned Fixed-Share/Beta confidence switched MTP→Markov→MTP without a stored phrase or replay; 24 output tokens in 14 target forwards, ten drafts accepted, 1.32 GB peak RSS |
| Position-specific Council memory | live v6→v7 migration; 16 feedback positions | unseen snow question trained position 0 exactly 15 times while leaving unobserved deeper positions at zero; Markov still owned one round; 24 output tokens in 16 target forwards; no synthetic horizon evidence |
| Online horizon growth | position observations `[15,0,…] → [28,3,0,…]` | unseen soap-bubble question; ordinary K2 carries supplied three exact position-1 observations, teacher replay remained zero because no wider mismatch suffix existed; 24 output tokens in 14 target forwards |
| Teacher-carry depth learning | position 1 observations `3 → 11` in one request | unseen autumn-leaves question; eight actual-prefix teacher predictions, seven confirmed by the following target token, zero replay failures; low measured position-1 accuracy correctly kept Markov from claiming that round |
| Causal Markov/MTP consensus | one shared-prefix agreement, zero unjustified gain | unseen salt-water question; Markov confidence below 0.5 kept consensus gain at exactly zero while eight teacher carries grew position-1 observations `11→20`; 24 tokens in 15 target forwards |
| Position-specialist consensus | 9 consensus rounds; 12 shared-prefix tokens | unseen sky question; position-specific experts contributed `+0.2095` confidence while MTP retained execution; position-1 observations reached 28 and specialist-weighted accuracy 32.55%; 24 tokens in 16 target forwards |
| Dialect × position specialists | 28 contextual specialist predictions | unseen silver-corrosion question matched the preceding iron-corrosion dialect at similarity 0.2549; dialect maturity 0.1622, two consensus tokens and positive gain; 24 tokens in 15 target forwards |
| Multi-dialect inference Council | 4 profiles; effective size 3.47 | unseen cloud question; active dialect plus three Ricci-ranked transfer neighbors jointly drove 28 contextual positions, three consensus rounds, four shared tokens and `+0.0425` confidence; 24 tokens in 15 target forwards |
| One-step Markov planning | 86 decisions; 344 bounded candidate evaluations | unseen thunder/lightning question; three greedy tokens changed, maximum sequence gain 0.8285, complete 20-token answer in 13 target forwards; forced phrases and position 15 bypass planning |
| Automatic Markov→MTP→Markov learning | 30.38 → 26.79 s; 12 → 9 target forwards | first novel request opened MTP and taught Council memory; after two confirmations the third request stayed entirely on a seven-token dialect phrase, loaded no MTP, preserved identical output, and used 1.34 GB peak RSS |
| In-request Markov→MTP handoff | 24 output tokens in 9 target forwards | four learned Markov K4 waves followed by one exact 41-row hidden-state handoff and four MTP waves; 15 accepted drafts; no Qwen replay; 29.93 s generation and 1.46 GB peak RSS |
| Compositional unseen token slot | unseen `CODE_DD44` via Markov only | three distinct target-confirmed code bindings induced unanimous Literal+Copy programs; fourth slot loaded no MTP, returned the exact requested code in three target forwards, and remained fully Qwen-verified |
| Rolling K=2–16 continuation | accepted prefixes commit with zero weight reads | one target-known token plus up to 15 drafts; DeltaNet Conv/recurrent state, next-token continuation, Graft, and native Prefix-Sinkhorn state remain bit-exact |
| Native-token Markov council | zero draft-model bytes | eight sparse Qwen-ID experts across orders 0–16; target-only Rapidity/Fixed-Share weighting, regime detection, 64 context dialects, Ricci retention, atomic episode learning, and target-confirmed variable phrase options up to 15 tokens |
| Bounded Atlas + online beam | at most eight retained paths | Atlas and live Council alternatives are fused per token, deterministically pruned, and calibrated only from reconciled target prefixes; K1 virtual matches, request-budget truncation, and EOS truncation update only proposals actually verified |
| Same-request high-order agent | repeated answer context becomes immediately eligible | order-2+ transitions learn from confirmed answer tokens and target corrections inside the active request; prompts and rejected drafts remain excluded, and the ephemeral agent adds no second persistence write |
| Same-request phrase/copy agent | up to 15 tokens from repeated confirmed spans | two or more prior occurrences induce their deterministic longest common continuation prefix; the active proposal width caps the copy, Full Q4 verifies it, and both request-local agents stay inside a 4,096-token horizon |
| Same-request Fixed-Share Council | expert winner changes before the next draft | every confirmed normal, K1-carry, external, and teacher-forced row updates an ephemeral Rapidity vector; abort discards it, while successful finalization replays each row exactly once into durable global/dialect state |
| Same-request position specialists | independent winners at recursive positions 0–15 | a provider-only 16×8 Beta overlay updates position weights, calibrated confidence, and language evidence immediately; finalization clears it before exact-once replay into durable global and dialect matrices |
| Same-request lookahead regret | harmful non-greedy plans stop inside the active answer | planned-vs-greedy hits are tracked independently at positions 0–15; the existing empirical log-advantage gate consumes the ephemeral outcomes immediately, while finalization persists each outcome exactly once |
| Corpus-scale Markov atlas | 4,000,000 tokens → 500,000 contexts in 8.0 MB | flat-array v2 adds ~150 MiB RSS and loads in 1.77 s on the live server; on an arbitrary rainbow question it supported 7 MTP tokens across 6 rounds while the hybrid accepted 11 drafts and produced 24 tokens in 14 target forwards |
| Live-answer Markov expert | 65,536 retained answer tokens; 4,096-token cached PPM windows | on an unseen German lightning paraphrase the online memory supported 64/98 MTP candidates and supplied 18 calibrated confidence gains across 12 rounds; 10 drafts accepted, 24 tokens in 15 target forwards |
| O1-valued episode retention | 1.1 KB state + 130 KB neural sidecar after the first live answer | real O1 surprise and learning-progress assign persistent episode priority; Ricci-age eviction keeps valuable older answers over low-value newer ones, with atomic rollback and answer-only boundaries |
| Persistent draft-window policy | request ceiling plus per-wave K4/K8/K16 | real Fixed-Share ceiling sampling after bootstrap; each Markov proposal supplies prefix-local expected acceptance/work utility and only the chosen prefix enters the target; full provider tails remain unauthoritative; model/tokenizer/provider-bound state; K2/K3 terminal fallback |
| Consuming ordinary decode | one continuation cache plus one replacement layer | removes simultaneous ownership of complete old and new cache stacks; official static cache cut is `154,927,104 + 65,552 × prefix_tokens` bytes |
| K4 Fast-MLP route reuse | repeated K2/K4 target and auxiliary bytes equal one route | Gate/Up rows are wave-unioned; one-route down cache uses maximum-overlap row ordering while preserving each reduction; identical K4 routes remove 180 MiB per active p4/k32 layer |
| Direct-to-Torch local ranges | one final tensor for sorted selected-row routes | inode-stable `preadv` fills caller-owned Torch storage; no intermediate Python body, no row-stack duplicate, exact cache/budget/causal-plan accounting |
| Markov range prefetch | zero additional logical/source bytes | grouped operations; order-1/2 beam; probability×Ricci/reuse scoring; cross-operation dedupe; accepted-only cooldown; delayed exact-byte utility for d1–d8 agents; adaptive 12.5–100% reservoir budget; bounded local OS-page hints |
| Weight-only all-layer Fast MLP | deterministic 64-layer p4/k32 plan; capture-only target use forbidden | no prompts or model forwards in the build; exact paths learn adaptive width plus prequential sparse-output cosine/L2 and scalar correction; non-beneficial or uncalibrated routes use the full MLP |
| Direct Q4 MLP-page route | explicit 192-of-272 page bound by default | exact full-MLP traces train bounded temporal/cross-layer/marginal routes; unready predictions fall back to full Q4, selected routes execute native Q4 pages, and speculative learning commits only for accepted rows; no wall-time claim is attached |
| Exact residual-PQ LM head | K=1–16 and k=1/3/7 parity; certified page + economic row pruning | deterministic weight-only build, best-first tree, actual row-code caps, K-query survivor union, contiguous-read cost gate, 64-leaf no-saving stop, outward residual/roundoff bounds, overflow/tie/subnormal guards, and exact full-leaf fallback; no model forward |
| Arbitrary local Qwen chat | one terminal 64-layer sweep removed per request | authenticated causal Qwen3.8 path; default zero-model-byte Markov drafting and full Q4 MLP execution, with auxiliary draft and MLP routes available explicitly; target, auxiliary, draft, and combined transport reported separately |
| Seed v3 native migration | exact parity on 8/8 inherited tensor outputs for Micro and 5M | SHA-first migration of shared trained GRU/CRSA/SwiGLU weights; new receipt, quotient, route-value, and expected-work heads added under a strict inference-only manifest |
| Full Qwen MLP layer map | 640/640 cells; content promotion 0 | every layer repeats the same 146/549 template hits; 1,370/1,389 candidate admissions equal their matched random controls, closing the exhaustive exact-key line |
| Qwen3.8 exact continuation | bit-identical K=1–16 state | exact transactional core with tokenwise hidden, KV, DeltaNet Conv/recurrent, and Prefix-Sinkhorn parity |
| Markov-OoE PoC | 96.10% PS-Lifted; 100% warm | 17.28% local baseline, 13.50% shuffled-Crystal placebo, 338→64 consensus rounds, 588-byte raw kernel payload |
| Consequence-grounded Markov language | 100% primitive, unseen-program, contextual-word, held-out grammar, option-word, cultural-transfer, and self-hosted execution | 5 seeds × 1,000 unseen programs; shuffled semantics 2.02%, fixed no-message policy 0.98%, no-action abstain 2.90%, shuffled grammar 5.00%, holistic held-out 0%, all in-vocabulary unknown words abstain |
| Recursive executable word DAG | 8,191 actions → 36 references → one Crystal; 8.380x fair VM speedup | 99.5605% reference reduction, 99.9878% deployment-symbol reduction, exact future-input parity, 1,638,000 historical work units released across 25 states |
| Continual executable-language growth | old actions 100% retained; promoted action learned in 328 episodes; final 100% | five seeds; initial promoted-action abstention 100%; no-memory 0%; context-schema shift keeps 100% global semantics and drops local evidence; reward-policy shift resets 4/4 actions |
| Second-generation compute word | one promoted macro reused twice → one charged Crystal | five seeds; constant discharge 5/5; historical work released rises from 400 to 1,200 units across 25 future states |
| Live O1 controller frontier | 40/40 probes → 200 observations → 8 promoted ComputeCrystals | official local Qwen bundle; exact controller snapshot, Atlas revision, model/weight pins, coverage, calibration, evidence, verifier, kernel, and append-only bank receipt |
| First live executable dialect | 100% frozen vocabulary; 1,000/1,000 greedy executions | eight real O1 site-policy actions; receiver reward comes only from replayed action/artifact/output parity |
| First live compiled word | 4 real actions → 1 charged Crystal; 3,375 work units released | three sealed support executions; 25 future vectors; maximum flat-versus-compiled delta 1.11e-16 |
| First live cross-dialect transfer | 8/8 surface words differ; semantic translation 8/8 | one supported four-action Qwen/O1 macro localizes into four sibling-native words, recompiles to one Crystal, releases 3,375 work units, and preserves parity within 2^-52 |
| Executable dialect mesh | semantic translation 60,000/60,000; worst target context 100% | five seeds × five dialects × three independently learned contexts; four target dialects have 3/3 distinct mappings and zero global action words; shared-vocabulary direct transfer 6.84%, permutation placebo 1.42% |
| Portable cross-dialect macro | localization and compilation 75/75; binding/authority changes rejected 5/5 | one discovered four-action program moves into every dialect-context without Q-table copying; context-bound exact execution; mean 150 historical work units released |
| Action-conditioned Markov planning | 250/250 unseen tasks | 68 one-step observations only; 4–10-step tasks; 12-replica local cohort 226/3,000 (7.53%), shuffled-action placebo 89/250, no-memory 68/250 |
| Continuous reservoir memory | 90.84% fused | delayed-state task; 82.16% local mean, 50.45% no-memory, 54.28% shuffled-label placebo |
| Compositional stored compute | 4 operators → 1 exact discharge | 128 unseen vectors; 27,648 historical work units released; maximum delta 1.78e-15 |
| Charged-prefix residual compute | 2 charged + 2 live suffix steps | 128 unseen vectors; 9,216 historical work units released; maximum delta 1.33e-15 |
| Birkhoff operator basis | 33 atoms; exact bound 50 | weighted atom discharge matches the stored 8×8 Markov operator within 6.66e-16 |
| Autonomous operator search | 12 MAP-Elites cells; late mutation preference 100% | 14 Pareto elites; Fiedler missing-bridge priority 0.409 vs. 0.121 for the best existing edge |
| Demand-routed residual execution | rewards 1.25 vs. 1.50; deeper prefix selected on decision 3 | persisted PPM/UCB selection → forced charged prefix → live suffix → external verifier → exact positive/negative feedback in one joined receipt |
| Exact guarded affine monoids | Stack depth 53/64 exact; `aⁿbⁿcⁿ` 4/4; placebos 7/7 rejected | exact integer/modular `(A,b)` atoms, guard-preserving fusion, 128-digit Horner, fingerprint collision contained by byte verifier, atomic replay-verified bundle |
| Contextual algebra agent | 120/120 correct; shuffled-context placebo 0/90 | Stack/Fingerprint/Decimal choice; Stack→Decimal regime recovery 30/30; three incompatible ABIs execute as one receipt-joined parallel ensemble |
| Harvested-program algebra routing | 174/180 correct; late 60/60 | three executable affine/permutation/Markov `ComputeProgram` families; shuffled-context placebo 49/180 and late 13/60; every result returns verifier-bound Thompson feedback |
| Causal sequence Crystal | exact future-mass zero; maximum fused delta 1.33e-15 | 32-dimensional, eight-operator randomized audit of left-acting causal kernels; dedicated fusion verifier and charge accounting |
| Contextual sequence predictor | MSE 6.73e-05 vs. tuned pointwise 5.80e-02 | untouched synthetic sequence holdout; 862.9x lower MSE; shuffled-token/output placebos 3.20/3.15; predictive-only artifact |
| Living O1/Qwen frontier | additive prompt × layer × site growth | immutable initial manifest plus append-only frontier events, exact scheduler reconciliation, persistent idle cycles, rollback rejection, and zero loss of Atlas/O1 history |
| Algebraic crystallization | 3/3 families admitted and exact | additive, multiplicative, and cyclic length-12 execution; permuted placebo rejected |
| Qwen compute battery | 1.4288x peak speed | 65-token charged prefix + unknown 33-token suffix; authenticated restore removes 30.01% of demand latency |
| O1 → Atlas → OoE | 10/10 live jobs; 15 measurements | two real contextual Qwen sites, Atlas revision 15, crash-resumable promotion, exact zero-probe reuse |
| Real Qwen context → operator harvest | 40/40 transitions grouped into 10 layer/sublayer families | four distinct prompts through the committed causal-Qwen fixture; whole, Attention-core/residual, and MLP-core/residual receipts remain separate and prompt-diverse |
| Real Qwen ResultCell holdout | 5 → 0 Qwen forwards | four temporal train transitions; holdout transition absent; exact raw-Qwen and final-semantic parity; gold-correct under one frozen evaluator call; FERTIG abstained on both paths |
| Qwen3.5 → Qwen3.8 live K=4 | 4/4 draft tokens accepted | fixed native Prefix-Sinkhorn trial; 32.39% fewer combined source bytes and 1.349x wall-time speedup versus 2×K=2 |
| Qwen GC pressure policy | bit-identical; 134 → 2 collections | fixed causal CPU-BF16 A/B; 6.88% less process wall time and 1.074x speedup |
| FERTIG exact frontier | 58/64 certified, 6 abstained | fixed gold-free development slice; holdout untouched |

The public benchmark ledger states the exact corpus and measurement boundary
for every released number: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Paper context

- David Tom Foss, **The `.causal` Format: Embedded Deterministic Inference for
  Domain-Agnostic Knowledge Graph Amplification**, IEEE IRI 2026. Conference
  record: [IEEE IRI session E2](https://davidtomfoss.com/service/iri2026-session-e2-nlp-sentiment-multimodal-reasoning/).
- David Tom Foss, **Deterministic Validation for Reliable LLM-Based Causal
  Knowledge Extraction**, ICECET 2026. Record:
  [davidtomfoss.com](https://davidtomfoss.com/talks/deterministic-validation-llm-causal-extraction/).
- Mathematical lineage and related peer-reviewed work:
  [docs/research.md](docs/research.md).
- Runtime equations for Sinkhorn attention, Markov transport, PS-Lifted world
  models, reservoirs, algebraic Crystals, stored compute, and novelty:
  [docs/FORMELN-LEVERAGE.md](docs/FORMELN-LEVERAGE.md).

## Related repositories

- [DT-Foss/FERTIG](https://github.com/DT-Foss/FERTIG)
- [DT-Foss/o1-state](https://github.com/DT-Foss/o1-state)
- [DT-Foss/dotcausal](https://github.com/DT-Foss/dotcausal)

## Citation and release boundary

Citation metadata is recorded in [CITATION.cff](CITATION.cff). The current
artifact version and public component map are recorded in
[CHANGELOG.md](CHANGELOG.md) and [manifests/components.json](manifests/components.json).

The release boundary is source-selective. It excludes frontier weights,
private datasets, learned route graphs, runtime traces, credentials, machine
topology, and private capability-transfer mechanisms.

## Author

[David Tom Foss](https://davidtomfoss.com/) ·
[ORCID 0009-0004-0289-7154](https://orcid.org/0009-0004-0289-7154)
