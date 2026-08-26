# Runtime Verdict

Release: 0.8.0 · 2026-08-26

## Verdict

IMMER is a working local causal-intelligence runtime with six integrated
execution planes:

1. local Qwen3.8 inference from immutable causalized weights with native Causal
   Prefix Sinkhorn Attention, exact continuation, and semantic state restore;
2. O1-State measurement and experiment selection over an append-only
   SemanticWeightAtlas;
3. replicated action-conditioned Markov world models, PS-Lifted consensus,
   hierarchical options, continuous reservoir memory, operator-demand
   learning, contextual algebra selection, and novelty admission;
4. generic stored computation through ComputeCrystals, an authenticated
   operator graph, contextual operator harvesting, Birkhoff atom bases,
   exact guarded affine monoids, algebraic crystallization, and demand-routed
   charged-prefix residual discharge;
5. deterministic FERTIG/S3 capability execution and sealed ResultCells under
   final parity accounting;
6. LiveCausal addressing and exact Safetensors range execution, with the
   DeepSeek-V4 line retained as the transport laboratory.

The architecture keeps the original model weights local and immutable. O1
measures them, Atlas names the evidence, Markov agents learn executable
structure, and authenticated crystals return previously paid computation to
later requests.

## Verified evidence

- Qwen continuation preserves complete tokenwise hidden, KV, DeltaNet, and
  Prefix-Sinkhorn state exactly across K=1–4.
- The first native Qwen compute battery reduces one fixed demand path from
  `130.457495 s` to `91.304167 s`, a `1.428823x` speedup with exact boundary
  parity.
- The original Markov-OoE experiment reaches `96.10%` PS-Lifted accuracy and
  `100%` warm execution against `17.28%` local and `13.50%` shuffled-Crystal
  controls.
- The action-conditioned world-model benchmark solves `250/250` unseen
  4–10-step tasks from 68 one-step observations. Across all 12 local replicas,
  local coverage solves `226/3,000 = 7.5333%`; the shuffled-action placebo
  solves `89/250`, and no-memory solves `68/250`.
- Hierarchical crystallization reduces one verified operator chain from depth
  eight to depth one. The structured MPO stores `1,408` numeric bytes instead
  of `8,192` at `3.58e-16` relative error; unsupported structure selects the
  exact dense representation.
- The continuous reservoir reaches `90.8359%` delayed-state accuracy against
  `82.1626%` local mean, `50.4532%` no-memory, and `54.2800%` shuffled-label
  controls.
- A generic four-operator ComputeCrystal route discharges as one operator on
  128 values created after charging, releases `27,648` authenticated historical
  work units, and matches primitive execution within `1.7764e-15`.
- A two-step charged prefix executes inside a four-step future route, computes
  only the two-step residual suffix, releases `9,216` historical work units,
  and matches primitive execution within `1.3323e-15`.
- The constructive 8×8 Birkhoff trial stores `33` permutation atoms under the
  exact bound `50` and reproduces the Markov operator within `6.6613e-16`.
  Behavioral MAP-Elites retains `14` Pareto operators across `12` cells; the
  Thompson mutation agent selects the successful arm on all `32` late trials.
- Fiedler-projector novelty ranks the strongest missing barbell bridge at
  `0.409362` against `0.120790` for the strongest existing edge after its
  direct-edge penalty.
- The joined demand executor explores two charged prefixes, verifies exact
  residual parity, receives rewards `1.25` and `1.50`, and selects the deeper
  prefix on decision three. Operational verifier failure produces a neutral
  abort event and leaves the arm's reward untouched.
- The exact stack recovers unique buried symbols at depth `53/64` and enters
  dead at depth `65`. The genuine `aⁿbⁿcⁿ` machine accepts `4/4` unseen counts
  through `n=100` and rejects `7/7` order/count placebos. Decimal-Horner is
  exact through 128 digits; the dual-modular collision stays behind the exact
  byte-verifier boundary.
- The contextual algebra agent chooses Stack, Fingerprint, and Decimal
  correctly on `120/120` decisions, scores `0/90` on the shuffled-context
  placebo, and recovers Stack→Decimal on `30/30` late decisions. Three
  incompatible program schemas execute as independent replay-verified lanes
  under one joined receipt.
- Four distinct prompts through the committed causal-Qwen fixture yield eight
  authenticated hidden transitions grouped into two reusable layer families.
  Exact prompt runtime receipts remain distinct and the stable operator family
  pools all four contexts.
- Additive, multiplicative, and finite-cyclic maps all pass admission above
  `0.999998` and execute length-12 compositions exactly; the permuted placebo
  is rejected.
- The current real local-Qwen cartography run completes `10/10` jobs and stores
  15 authenticated measurements across two sites at Atlas revision 15.
- The real ResultCell holdout reduces an authenticated five-forward Qwen
  baseline to zero warm forwards and commits all five as saved. Four earlier
  rows supply the teacher transitions; the holdout transition is absent.
  Raw-Qwen documents and final semantic cores match exactly, and one frozen
  evaluator call verifies the gold-correct output.
- FERTIG scores 1,089 correct, zero wrong, and 230 abstentions across all 1,319
  GSM8K test rows. SHIP-v6 returns 152/152 answers and 152/152 routes.

Every number above is bounded to the named receipt, corpus, topology, or fixed
benchmark. [BENCHMARKS.md](BENCHMARKS.md) records the exact boundary and public
evidence surface for each result.

## Runtime contracts

| Plane | Implementation | Contract |
|---|---|---|
| local Qwen | `src/immer/runtimes/qwen3_8/` | authenticated causal mount, exact stateful inference, Prefix-Sinkhorn, anchors |
| O1 and Atlas | `src/immer/runtimes/o1_state/`, `src/immer/runtimes/ooe/cartography.py` | persistent learning state, exact measurement identity, append-only revision membership |
| Markov intelligence | `src/immer/runtimes/ooe/` | action-conditioned evidence, PS-Lifted fusion, planning, options, demand execution, contextual algebra routing, novelty, topology and regime receipts |
| Stored operators | `compute_crystals.py`, `compute_graph.py`, `residual_execution.py`, `demand_execution.py`, `operator_harvester.py`, `bvn_search.py`, `bvn_crystals.py`, `affine_monoid.py`, `algebra_agents.py`, `algebraic_crystals.py` | real contextual discovery, typed unseen-input programs, constructive atom bases, exact integer/modular recurrence, authenticated charge and feedback, heterogeneous parallel execution, atomic persistence, exact accounting |
| ResultCell | `result_cells.py` | one complete cold-result binding, zero-forward warm execution, final semantic and evaluator parity |
| exact execution | `src/immer/cognition/` | S3/FERTIG verification, grounded fallback, explicit abstention |
| LiveCausal | `src/immer/knowledge/livecausal.py` | immutable payloads, append-only causal graph, lazy exact queries, recovery |

`ComputeCrystal` and `ResultCell` solve different problems. A ComputeCrystal is
a general operator that accepts new runtime values. A ResultCell is an exact
sealed endpoint for one cold Qwen/FERTIG execution identity. Both persist
atomically; only ComputeCrystal provides value-general operator reuse.

## Real ResultCell boundary

The completed cold-to-warm cohort contains five authenticated cold cells with
forward counts `[6, 7, 7, 6, 5]`. Exactly four temporal teacher transitions are
ingested. The fifth transition remains absent while the learned action executes
the holdout with zero Qwen forwards. Shuffled-site and shuffled-Crystal controls
remain non-executing, both stores audit clean, and exact resume repeats none of
the five cold generations.

The holdout's cold and warm FERTIG judgments both abstain. Therefore
`fertig_exact_judgment=false` and `fertig_semantic_certified=false`. The result
is verified by the frozen evaluator's gold-correct decision, opened exactly
once, together with exact raw-Qwen-document and final-semantic-core parity.

## Public release boundary

Public source contains algorithms, interfaces, tests, manifests, and selected
small benchmark receipts. Model weights, learned live graphs, private prompts,
route traces, deployment topology, graft payloads, credentials, and capability
transfer artifacts remain outside Git.
