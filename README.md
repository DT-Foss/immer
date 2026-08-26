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
  weights with stateful full-attention and DeltaNet continuation.
- **Native local drafting.** A causalized Qwen3.5-0.8B proposes transactional
  K=2 and K=4 continuations that Qwen3.8 alone verifies and commits.
- **Grounded composition.** [FERTIG](https://github.com/DT-Foss/FERTIG)
  supplies deterministic parsing, verification, and explicit abstention.
- **Organism of Experts.** O1 measurements become authenticated Atlas evidence;
  PS-Lifted Markov agents learn action-conditioned world models, compose
  verified options, harvest real contextual Qwen transitions, and publish
  executable Crystals while Qwen supplies cold evidence and handles genuine
  novelty.
- **Compute batteries.** Native Qwen continuation states are charged ahead of
  demand, restored by exact token prefix, and resumed without replaying the
  authenticated prefix.
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
| Qwen3.8 exact continuation | bit-identical K=1–4 state | exact transactional core with tokenwise hidden, KV, DeltaNet, and Prefix-Sinkhorn parity |
| Markov-OoE PoC | 96.10% PS-Lifted; 100% warm | 17.28% local baseline, 13.50% shuffled-Crystal placebo, 338→64 consensus rounds, 588-byte raw kernel payload |
| Action-conditioned Markov planning | 250/250 unseen tasks | 68 one-step observations only; 4–10-step tasks; 12-replica local cohort 226/3,000 (7.53%), shuffled-action placebo 89/250, no-memory 68/250 |
| Continuous reservoir memory | 90.84% fused | delayed-state task; 82.16% local mean, 50.45% no-memory, 54.28% shuffled-label placebo |
| Compositional stored compute | 4 operators → 1 exact discharge | 128 unseen vectors; 27,648 historical work units released; maximum delta 1.78e-15 |
| Charged-prefix residual compute | 2 charged + 2 live suffix steps | 128 unseen vectors; 9,216 historical work units released; maximum delta 1.33e-15 |
| Birkhoff operator basis | 33 atoms; exact bound 50 | weighted atom discharge matches the stored 8×8 Markov operator within 6.66e-16 |
| Autonomous operator search | 12 MAP-Elites cells; late mutation preference 100% | 14 Pareto elites; Fiedler missing-bridge priority 0.409 vs. 0.121 for the best existing edge |
| Demand-routed residual execution | rewards 1.25 vs. 1.50; deeper prefix selected on decision 3 | persisted PPM/UCB selection → forced charged prefix → live suffix → external verifier → exact positive/negative feedback in one joined receipt |
| Exact guarded affine monoids | Stack depth 53/64 exact; `aⁿbⁿcⁿ` 4/4; placebos 7/7 rejected | exact integer/modular `(A,b)` atoms, guard-preserving fusion, 128-digit Horner, fingerprint collision contained by byte verifier, atomic replay-verified bundle |
| Contextual algebra agent | 120/120 correct; shuffled-context placebo 0/90 | Stack/Fingerprint/Decimal choice; Stack→Decimal regime recovery 30/30; three incompatible ABIs execute as one receipt-joined parallel ensemble |
| Algebraic crystallization | 3/3 families admitted and exact | additive, multiplicative, and cyclic length-12 execution; permuted placebo rejected |
| Qwen compute battery | 1.4288x peak speed | 65-token charged prefix + unknown 33-token suffix; authenticated restore removes 30.01% of demand latency |
| O1 → Atlas → OoE | 10/10 live jobs; 15 measurements | two real contextual Qwen sites, Atlas revision 15, crash-resumable promotion, exact zero-probe reuse |
| Real Qwen context → operator harvest | 8/8 transitions grouped into 2 layer families | four distinct prompts through the committed causal-Qwen fixture; exact runtime receipts remain separate while stable families pool correctly |
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
