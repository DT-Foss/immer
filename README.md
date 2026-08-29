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
- **Native local drafting.** A causalized Qwen3.5-0.8B or a zero-weight Markov
  council proposes transactional continuations in a configurable K=2–16
  target window. Qwen3.8 alone verifies and commits them; K=8 is the product
  default. Target-confirmed episodes become reusable three-token options;
  global and contextual-dialect agents compete by support, confidence, context
  depth, and dialect similarity.
- **All-layer Fast MLP.** A prompt-free weight initializer derives p4/k32
  pilots for every Qwen MLP layer from exact Gate/Up Gaussian joint moments.
  Existing full-path tensors update bounded recursive-ridge route statistics;
  confidence, width, warmup, and periodic checks select the unchanged full
  MLP without adding a model forward. [All-layer runtime](docs/FAST-MLP-ALL-LAYER.md).
- **Exact causal LM-head rail.** A weight-only residual-PQ tree supplies
  certified upper bounds for canonical token pages. Proven-impossible pages
  require no checkpoint read; every unresolved page uses the same explicit
  FP32-accumulate/BF16-output scorer and stable lower-token-ID tie rule.
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
  system.
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

## Selected trial evidence

| Trial | Result | Scope |
|---|---:|---|
| Local causal bundle reopen | 77.77 → 1.20 s; 64.90x | complete 55.6 GB Qwen3.8 bundle; first full-content verification followed by unchanged next-process stat+digest reuse; no model forward |
| Rolling K=2–16 continuation | accepted prefixes commit with zero weight reads | one target-known token plus up to 15 drafts; DeltaNet Conv/recurrent state, next-token continuation, Graft, and native Prefix-Sinkhorn state remain bit-exact |
| Native-token Markov council | zero draft-model bytes | eight sparse Qwen-ID experts across orders 0–16; target-only Rapidity/Fixed-Share weighting, regime detection, 64 context dialects, Ricci retention, atomic episode learning, and target-confirmed three-token phrase options |
| Consuming ordinary decode | one continuation cache plus one replacement layer | removes simultaneous ownership of complete old and new cache stacks; official static cache cut is `154,927,104 + 65,552 × prefix_tokens` bytes |
| K4 Fast-MLP route reuse | repeated K2/K4 target and auxiliary bytes equal one route | Gate/Up rows are wave-unioned; one-route down cache uses maximum-overlap row ordering while preserving each reduction; identical K4 routes remove 180 MiB per active p4/k32 layer |
| Direct-to-Torch local ranges | one final tensor for sorted selected-row routes | inode-stable `preadv` fills caller-owned Torch storage; no intermediate Python body, no row-stack duplicate, exact cache/budget/causal-plan accounting |
| Weight-only all-layer Fast MLP | deterministic 64-layer p4/k32 plan; exact-path online learning | no prompts or model forwards in the build; one/reused route addresses 18.0147% of MLP rows; fixed-size target-confirmed state and full-MLP fallback |
| Exact residual-PQ LM head | K=1–16 and k=1/3/7 parity; certified page pruning | deterministic weight-only build, best-first tree, outward residual/roundoff bounds, overflow/tie/subnormal guards, and one-pass exact fallback; no model forward |
| Arbitrary local Qwen chat | one terminal 64-layer sweep removed per request | authenticated causal Qwen3.8 path; rolling local-Qwen or zero-model-byte Markov drafting plus row-routed fast MLP; target, auxiliary, draft, and combined transport reported separately |
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
