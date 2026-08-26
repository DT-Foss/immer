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
   hierarchical options, continuous reservoir memory, and novelty admission;
4. generic stored computation through ComputeCrystals, an authenticated
   operator graph, algebraic crystallization, and exact warm discharge;
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
| Markov intelligence | `src/immer/runtimes/ooe/` | action-conditioned evidence, PS-Lifted fusion, planning, options, novelty, topology and regime receipts |
| ComputeCrystal | `compute_crystals.py`, `compute_graph.py`, `algebraic_crystals.py` | typed unseen-input programs, authenticated charge basis, atomic persistence, exact discharge accounting |
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
