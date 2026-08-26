# Benchmarks

Release: Unreleased · 2026-08-26

Each row states its exact measurement scope and evidence boundary. Committed
machine-readable receipts are linked directly. Model-weight execution receipts
that remain private are identified as private and paired with the committed
harness and validation tests that define their schema. Transport
microbenchmarks are separated from model-quality benchmarks, and
remote-network results are separated from local execution.

## Runtime and quality

| Benchmark | Result | Scope | Evidence |
|---|---:|---|---|
| SHIP-v6 exact suite | 152/152 answers; 152/152 routes | frozen host plus four exact organs | `manifests/s3_ship_v6.json`; `python -m immer eval` |
| FERTIG GSM8K | 1,089 correct; 0 wrong; 230 abstentions; 82.56% coverage | all 1,319 GSM8K test rows, certificate-first and strict zero-wrong scoring | [`results/bench_gsm8k.json`](../results/bench_gsm8k.json) |
| FERTIG exact Dev frontier | 58/64 certified; 6 abstained | fixed gold-free question-only development slice; structural path 57/64 plus one guarded-formula certificate; holdout remains untouched; selection receipt private | [`signed_event_frontend.py`](../src/immer/cognition/fertig/signed_event_frontend.py); [`discourse_ssa.py`](../src/immer/cognition/fertig/discourse_ssa.py); [`test_fertig_bundle_tariff_algebra.py`](../tests/test_fertig_bundle_tariff_algebra.py) |
| FERTIG abstention recovery | 74/230 recovered; 156 remain abstained | gold-free cumulative chain: 42 sealed baseline + 5 grounded operators + 10 typed discourse SSA + 9 schedule + 4 recurrence + 4 bundle/tariff | [`results/fertig-abstention-audit.json`](../results/fertig-abstention-audit.json); [`fertig_abstention_audit.py`](../scripts/fertig_abstention_audit.py) |
| DeepSeek V4 MMLU | 3/4 correct; 43/43 layers complete | exact `off` mode on one fixed four-item high-school-geography slice | [`results/deepseek-v4-mmlu-off-4-exact-v4-window2x3-receipt.json`](../results/deepseek-v4-mmlu-off-4-exact-v4-window2x3-receipt.json) |
| DeepSeek V4 + FERTIG | 6/8 correct; 0 wrong; 2 abstentions | fixed eight-item GSM8K failure slice; five exact certificates plus one model-verified answer | [`results/deepseek-v4-fertig-fusion.json`](../results/deepseek-v4-fertig-fusion.json) |
| Qwen3.8 + FERTIG | 5/8 exact; 0 wrong; 3 quarantined | fixed eight-item GSM8K failure slice; model agreement cannot answer without an independent certificate | [`results/qwen38_fertig_fusion.json`](../results/qwen38_fertig_fusion.json) |

The four-item and eight-item rows are integration slices. Their claims stop at
those exact item sets.

## Markov-OoE and compute batteries

| Benchmark | Result | Exact scope | Evidence boundary |
|---|---:|---|---|
| Standalone Markov-OoE | local `17.28%`; reversible `49.84%`; PS-Lifted `96.10%`; warm `100%`; shuffled Crystal `13.50%` | six finite operator families; local noisy one-step teacher transitions; 250 unseen plans and 5,000 longer unseen compositions | imported package manifest `28/28`; production port and regression tests in [`runtimes/ooe`](../src/immer/runtimes/ooe) |
| PS-Lifted consensus | `338 → 64` rounds (`5.28125x` fewer) | original fixed barbell topology and tolerance; raw six-kernel payload `588 B` | [`test_ooe_math_consensus.py`](../tests/test_ooe_math_consensus.py); [`test_ooe_qwen_bridge_controller.py`](../tests/test_ooe_qwen_bridge_controller.py) |
| Action-conditioned world model | PS-Lifted `250/250`; central table `250/250`; 12-replica local cohort `226/3,000 = 7.5333%`; shuffled-action placebo `89/250`; no-memory `68/250` | 17 states, four actions, 12 replicas, 68 authenticated one-step observations, 250 unseen 4–10-step tasks per policy evaluation, zero multi-step labels | [`intelligence.py`](../src/immer/runtimes/ooe/intelligence.py); [`ooe_intelligence_benchmark.py`](../scripts/ooe_intelligence_benchmark.py); [`test_ooe_intelligence_benchmark.py`](../tests/test_ooe_intelligence_benchmark.py) |
| Markov self-calibration and topology shift | fixed-`pc` barbell `96` rounds → self-calibrated barbell `58`; complete topology `28`; success `100%` throughout; maximum fused-count delta `6.02e-9` | same fixed-seed world-model trial; self-calibration changes the consensus envelope, then topology changes after evidence collection | [`consensus.py`](../src/immer/runtimes/ooe/consensus.py); [`test_ooe_math_consensus.py`](../tests/test_ooe_math_consensus.py) |
| Hierarchical options | primitive depth `8 → 1`; success probability `1.0` | seven discovered options from four verified trajectories; exact contraction ledger; no multi-step teacher labels | [`options.py`](../src/immer/runtimes/ooe/options.py); [`contraction_ledger.py`](../src/immer/runtimes/ooe/contraction_ledger.py); [`test_ooe_options_mpo.py`](../tests/test_ooe_options_mpo.py) |
| Continuous PS-Lifted reservoir | fused `90.8359%`; local mean `82.1626%`; no-memory `50.4532%`; shuffled-label placebo `54.2800%` | eight replicas, 32-node fixed reservoir, 500 training steps per replica, 1,000-step test, delayed-state target at lag seven | [`reservoir.py`](../src/immer/runtimes/ooe/reservoir.py); [`reservoir_intelligence.py`](../src/immer/runtimes/ooe/reservoir_intelligence.py); [`test_ooe_reservoir_intelligence.py`](../tests/test_ooe_reservoir_intelligence.py) |
| Generic ComputeCrystal discharge | four primitive operators → one live operator; `27,648` authenticated historical work units released; maximum delta `1.7764e-15` | route charged before 128 unseen six-dimensional inputs existed; exact reopen/replay; disconnected placebo abstains | [`compute_crystals.py`](../src/immer/runtimes/ooe/compute_crystals.py); [`compute_graph.py`](../src/immer/runtimes/ooe/compute_graph.py); [`test_ooe_crystal_intelligence.py`](../tests/test_ooe_crystal_intelligence.py) |
| Charged-prefix residual discharge | two charged prefix steps + two live suffix steps; `9,216` historical work units released; maximum delta `1.3323e-15` | same 128 unseen six-dimensional inputs; prefix was charged before inputs existed; suffix remains primitive and live; uncharged materialized routes receive zero reuse credit | [`residual_execution.py`](../src/immer/runtimes/ooe/residual_execution.py); [`test_ooe_residual_execution.py`](../tests/test_ooe_residual_execution.py); [`test_ooe_crystal_intelligence.py`](../tests/test_ooe_crystal_intelligence.py) |
| Constructive Birkhoff Crystal basis | `33` atoms under exact bound `50`; maximum weighted-basis delta `6.6613e-16` | fixed seeded 8×8 doubly-stochastic operator; O(Kn) atom storage; 96 future inputs; full basis, receipt, reopen, inverse-permutation semantics, idempotence, and non-DS rejection tested | [`bvn_search.py`](../src/immer/runtimes/ooe/bvn_search.py); [`bvn_crystals.py`](../src/immer/runtimes/ooe/bvn_crystals.py); [`test_ooe_bvn_search.py`](../tests/test_ooe_bvn_search.py); [`test_ooe_bvn_crystals.py`](../tests/test_ooe_bvn_crystals.py) |
| Autonomous operator search | `12/64` MAP-Elites cells occupied; `14` Pareto elites; preferred mutation arm `32/32` late choices | 64 constructive DS candidates; three mutation families; fixed seed; Fiedler missing-bridge priority `0.409362` versus `0.120790` for the highest existing edge after direct-edge penalty | [`bvn_search.py`](../src/immer/runtimes/ooe/bvn_search.py); [`crystal_intelligence.py`](../src/immer/runtimes/ooe/crystal_intelligence.py); [`test_ooe_crystal_intelligence.py`](../tests/test_ooe_crystal_intelligence.py) |
| Verified operator demand | unseen-first UCB1; exact negative-edge blocking; selection-ordered PPM; episode-scoped co-occurrence; byte-exact Ricci retention | deterministic contract suite with asynchronous outcome settlement, standalone-event placebo, ABI/revision mismatch, forged semantic receipt, concurrent CAS, crash, rollback, fork, tamper, pins, and full program/Crystal/charge byte accounting | [`demand_scheduler.py`](../src/immer/runtimes/ooe/demand_scheduler.py); [`test_ooe_demand_scheduler.py`](../tests/test_ooe_demand_scheduler.py) |
| Demand-routed production loop | prefix rewards `1.25` and `1.50`; decision three selects the deeper prefix | same unseen 128-input affine route; exact chain PPM/UCB → forced prefix → residual VM → `1e-12` parity verifier → persisted outcome; operational verifier failure produces neutral abort and zero reward poisoning | [`demand_execution.py`](../src/immer/runtimes/ooe/demand_execution.py); [`test_ooe_demand_execution.py`](../tests/test_ooe_demand_execution.py); [`test_ooe_crystal_intelligence.py`](../tests/test_ooe_crystal_intelligence.py) |
| Exact guarded affine monoids | stack depth `53/64` exact and depth `65` dead; `aⁿbⁿcⁿ` `4/4`; placebos `7/7`; Decimal 128 digits exact | integer/per-coordinate modular `(A,b)` actions; guard-preserving fusion; explicit phase/dead/capacity/bit bounds; deliberate modular fingerprint collision becomes candidate and fails exact byte verification; one atomic replay-verified bundle | [`affine_monoid.py`](../src/immer/runtimes/ooe/affine_monoid.py); [`affine_intelligence.py`](../src/immer/runtimes/ooe/affine_intelligence.py); [`test_ooe_affine_monoid.py`](../tests/test_ooe_affine_monoid.py); [`test_ooe_affine_intelligence.py`](../tests/test_ooe_affine_intelligence.py) |
| Contextual algebra meta-agent | correct `120/120`; shuffled-context placebo `0/90`; Stack→Decimal recovery `30/30` | three verified Stack/Fingerprint/Decimal programs, contextual Thompson updates, three-cell MAP-Elites catalog, exact restart; handcrafted selections and stale verifier outcomes rejected | [`algebra_agents.py`](../src/immer/runtimes/ooe/algebra_agents.py); [`affine_intelligence.py`](../src/immer/runtimes/ooe/affine_intelligence.py); [`test_ooe_algebra_agents.py`](../tests/test_ooe_algebra_agents.py) |
| Harvested executable algebra routing | contextual `174/180`, late `60/60`, final `30/30`; shuffled-context placebo `49/180`, late `13/60`, final `5/30` | three held-out affine/permutation/Markov promotions converted to tagged `ComputeProgram` candidates; exact VM execution and verifier feedback on every decision; fixed seed `20260826` | [`harvest_algebra_bridge.py`](../src/immer/runtimes/ooe/harvest_algebra_bridge.py); [`harvest_algebra_intelligence.py`](../src/immer/runtimes/ooe/harvest_algebra_intelligence.py); [`ooe_harvest_algebra_benchmark.py`](../scripts/ooe_harvest_algebra_benchmark.py); [`test_ooe_harvest_algebra_intelligence.py`](../tests/test_ooe_harvest_algebra_intelligence.py) |
| Exact parallel-route choice | requested route SHA survives plan, charge, reopen, and discharge | two charged affine routes share the same source and goal but produce distinct outputs; `discharge_exact()` executes each requested route while canonical discharge retains cost-based selection | [`compute_graph.py`](../src/immer/runtimes/ooe/compute_graph.py); [`test_ooe_compute_graph.py`](../tests/test_ooe_compute_graph.py) |
| Heterogeneous exact ensemble | three independent ABIs, one joined receipt, replay exact | Decimal, Fingerprint, and Stack execute in parallel; each lane embeds program and initial state, replays standalone, and rejects swapped, missing, cross-program, same-schema, or verifier-artifact splices | [`algebra_agents.py`](../src/immer/runtimes/ooe/algebra_agents.py); [`test_ooe_algebra_agents.py`](../tests/test_ooe_algebra_agents.py) |
| Algebraic Crystals | additive `0.999999985`, multiplicative `0.999999844`, cyclic `0.999998261`; all length-12 results exact | nine support points per continuous family, 12 cyclic supports, unseen length-12 composition; permuted placebo score `0.155535` and rejected | [`algebraic_crystals.py`](../src/immer/runtimes/ooe/algebraic_crystals.py); [`test_ooe_algebraic_crystals.py`](../tests/test_ooe_algebraic_crystals.py) |
| Structured MPO kernel | `1,408 / 8,192` numeric bytes (`0.171875`); relative error `3.58e-16` | structured action-conditioned tensor; unstructured tensor selects authenticated exact-dense fallback when rank budget is exceeded | [`mpo.py`](../src/immer/runtimes/ooe/mpo.py); [`test_ooe_options_mpo.py`](../tests/test_ooe_options_mpo.py) |
| Hopfield novelty gate | `2/2` in-distribution accepted; `0/3` out-of-distribution false accepts | calibrated per-site energy and gap thresholds; removing the novelty gate accepts all three OOD cases | [`novelty.py`](../src/immer/runtimes/ooe/novelty.py); [`test_ooe_novelty.py`](../tests/test_ooe_novelty.py) |
| Qwen native anchor battery | `130.457495 s → 91.304167 s`; `1.428823x`; `30.012326%` demand latency removed | one official local CPU-BF16 Qwen3.8 cell; 65-token charged invariant and previously unknown 33-token suffix | private sealed receipt; [`anchor_battery.py`](../src/immer/runtimes/qwen3_8/anchor_battery.py); [`test_qwen3_8_adapter.py`](../tests/test_qwen3_8_adapter.py) |
| O1 → Atlas → OoE integration | live `10/10` jobs, `15` measurements, two sites, Atlas revision `15`, `501.95 s`; zero duplicate probes on Atlas reuse | five frozen public-GSM8K prompt identities; real contextual local-Qwen measurements at passive layer 0 and native Sinkhorn layer 27 | [`qwen38_o1_cartography.py`](../scripts/qwen38_o1_cartography.py); [`test_qwen38_o1_cartography.py`](../tests/test_qwen38_o1_cartography.py) |
| Living O1/Qwen frontier | combined cartography contracts `52/52`; exact append/resume/rollback/crash behavior | deterministic prompt × layer × site × intervention expansion; journal append leaves initial manifest immutable; scheduler preserves all O1/Atlas history; idle runner resumes and consumes later work | [`qwen38_o1_cartography.py`](../scripts/qwen38_o1_cartography.py); [`cartographer.py`](../src/immer/runtimes/o1_state/cartographer.py); [`test_qwen38_o1_cartography.py`](../tests/test_qwen38_o1_cartography.py); [`test_o1_cartographer.py`](../tests/test_o1_cartographer.py) |
| Real Qwen contextual operator stream | `8/8` projected transitions accepted into two layer families; four observations per family | four distinct prompts through the committed tiny causal-Qwen runtime; two measured layers per prompt; exact per-prompt runtime receipts remain distinct; stable model/code/dependency/platform/weight identity pools the compatible contexts | [`cartography_probe.py`](../src/immer/runtimes/qwen3_8/cartography_probe.py); [`operator_harvester.py`](../src/immer/runtimes/ooe/operator_harvester.py); [`qwen38_o1_cartography.py`](../scripts/qwen38_o1_cartography.py); [`test_qwen3_8_cartography_probe.py`](../tests/test_qwen3_8_cartography_probe.py) |
| Real Qwen ResultCell holdout | authenticated baseline `5` Qwen forwards → warm `0`; saved `5/5`; exact raw-Qwen document and final-semantic parity; evaluator-quality verified; gold-correct | five cold cells with counts `[6,7,7,6,5]`; four temporal teacher transitions; fifth-row transition absent; evaluator opens exactly once; FERTIG abstains cold and warm | private sealed result and verification receipts; [`qwen38_ooe_chat_cohort.py`](../scripts/qwen38_ooe_chat_cohort.py); [`test_qwen38_ooe_chat_cohort.py`](../tests/test_qwen38_ooe_chat_cohort.py) |
| ResultCell contract | zero-forward exact replay; forged baseline, mixed provenance, stale binding, prompt leakage, crash, and payload tamper rejected | synthetic and authenticated-fixture contract tests only; complete cold Qwen/FERTIG binding and final parity gate | [`result_cells.py`](../src/immer/runtimes/ooe/result_cells.py); [`test_ooe_result_cells.py`](../tests/test_ooe_result_cells.py) |

The warm controller counts a saved Qwen forward only after an execution-bound
result passes its final verifier. Anchor restoration reports prefix token-layer
work and checkpoint reads separately; Atlas reuse reports avoided probe calls;
ComputeCrystal discharge reports released operator work. These counters measure
different execution planes and remain separate.

`ComputeCrystal` is the general unseen-input operator path. `ResultCell` is the
exact-binding path for a complete cold Qwen/FERTIG result. The real holdout
above establishes zero-forward replay under exact raw-document parity, exact
final-semantic parity, and one frozen evaluator call. Its FERTIG certificate
flags are false because both paths abstain; its verified quality authority is
the frozen evaluator plus exact parity.

The shuffled-site and shuffled-Crystal controls both remain non-executing. The
Crystal and ResultCell stores audit clean. An exact rerun resumes from all five
durable cold cells and performs no repeated cold Qwen generation.

## Local causal Qwen continuation

| Benchmark | Result | Exact scope | Evidence boundary |
|---|---:|---|---|
| Qwen3.8 exact K=2 continuation | state and hidden bit-identical; `49.999989%` fewer source bytes; `46.483087%` less model time; `1.868568x` speedup | one fixed `[token, EOS]` continuation, two fresh 64-layer CPU-BF16 causal-bundle runtimes; all KV and DeltaNet state hashes equal | private sealed receipt; [`qwen38_continuation_block_parity.py`](../scripts/qwen38_continuation_block_parity.py); [`test_qwen38_continuation_block_parity.py`](../tests/test_qwen38_continuation_block_parity.py) |
| Qwen3.8 fixed-Q3 speculative K=2 | 2/2 accepted; `33.893734%` fewer target source bytes; `22.435086%` less model time; `1.289243x` speedup | one fixed gold-free Q3 pair; greedy and speculative tokens, hidden, cursor, KV, and DeltaNet state equal | private sealed receipt; [`qwen38_speculative_k2_benchmark.py`](../scripts/qwen38_speculative_k2_benchmark.py); [`test_qwen38_speculative_k2_benchmark.py`](../tests/test_qwen38_speculative_k2_benchmark.py) |
| Live Qwen3.5-0.8B → Qwen3.8-27B K=2 | 2/2 accepted; `31.244579%` fewer combined target-plus-drafter source bytes; `23.029270%` less wall time; `1.299195x` speedup | one fixed two-token, same-prompt CPU-BF16 native Prefix-Sinkhorn trial; target output `[760, 6511]`; target alone commits | private sealed live receipt plus an immediate same-runtime greedy control; [`qwen35_live_k2_smoke.py`](../scripts/qwen35_live_k2_smoke.py); [`test_qwen35_live_k2_smoke.py`](../tests/test_qwen35_live_k2_smoke.py); [`test_qwen35_local_draft.py`](../tests/test_qwen35_local_draft.py) |
| Live Qwen3.5-0.8B → Qwen3.8-27B K=4 | 4/4 accepted; `33.893858%` fewer target bytes; `32.391019%` fewer combined bytes; `25.885088%` less wall time; `1.349256x` speedup | one fixed four-token CPU-BF16 native Prefix-Sinkhorn trial versus same-prompt 2×K=2; output, target/drafter final state, CRSA history, and all four hidden positions equal | private sealed K=4 and 2×K=2 receipts; [`qwen35_live_k4_smoke.py`](../scripts/qwen35_live_k4_smoke.py); [`test_qwen35_live_k4_smoke.py`](../tests/test_qwen35_live_k4_smoke.py) |
| Qwen3.8 GC pressure policy | bit-identical; collections `134 → 2`; `6.875829%` less process wall time; `1.073835x` speedup | fixed causal CPU-BF16 prefix-plus-K2 native Prefix-Sinkhorn A/B; payload bytes and model math unchanged | private sealed A/B receipts; [`pager.py`](../src/immer/runtimes/qwen3_8/pager.py); [`test_qwen3_8_config_pager.py`](../tests/test_qwen3_8_config_pager.py); [`test_qwen3_8_model.py`](../tests/test_qwen3_8_model.py) |

The live trial compares `151,204,957,184` greedy target bytes with
`99,955,757,056` live target bytes plus `4,005,847,104` drafter bytes. Wall
time changes from `243.376159193 s` to `187.328407226 s`. This is a positive
mechanism and fixed-prompt speed result, not a cohort acceptance or quality
claim.

The K=4 trial compares two accepted K=2 rounds with one accepted K=4 round.
Target bytes change from `151,204,977,664` to `99,955,777,536`; combined
target-plus-drafter bytes change from `158,220,413,376` to `106,971,209,152`;
wall time changes from `261.195523944 s` to `193.584833717 s`. The exact core
supports K=1–4; this receipt measures one fixed K=4 prompt.

The committed four-prompt long-lived cohort harness is the next measurement
gate. It freezes question-only selection, counterbalanced K4/2×K2 ordering,
exactly-once bundle authentication, and full target-state parity. Execution is
pending and this ledger reports no cohort result:
[`qwen35_k4_cohort.py`](../scripts/qwen35_k4_cohort.py),
[`test_qwen35_k4_cohort.py`](../tests/test_qwen35_k4_cohort.py).

Run the committed contract tests:

```bash
PYTHONPATH=src:tests python -m unittest \
  test_qwen38_continuation_block_parity \
  test_qwen38_speculative_k2_benchmark \
  test_qwen35_local_draft \
  test_qwen35_live_k2_smoke \
  test_qwen35_live_k4_smoke \
  test_qwen35_k4_cohort \
  test_qwen3_8_speculative \
  test_qwen3_8_config_pager
```

The live harness accepts explicit causal-bundle paths and writes one sealed
result:

```bash
PYTHONPATH=src python scripts/qwen35_live_k2_smoke.py \
  --target-bundle <qwen3.8-causal-bundle> \
  --draft-bundle <qwen3.5-causal-bundle> \
  --prompt 'The capital of France is' \
  --max-new-tokens 2 \
  --attention-mode native-crsa \
  --device cpu \
  --compute-dtype bfloat16 \
  --output <result.json>

PYTHONPATH=src python scripts/qwen35_live_k4_smoke.py \
  --target-bundle <qwen3.8-causal-bundle> \
  --draft-bundle <qwen3.5-causal-bundle> \
  --prompt 'The capital of France is' \
  --max-new-tokens 4 \
  --attention-mode native-crsa \
  --device cpu \
  --compute-dtype bfloat16 \
  --parity-control tokenwise \
  --expected-token-ids 760,6511,314,9338 \
  --output <result.json>
```

## Causal Prefix Sinkhorn Attention and routing

| Measurement | Result | Scope |
|---|---:|---|
| CRSA context router | 182/182 arithmetic/text decisions | fixed router evaluation corpus |
| Raw frozen-host ablation | balanced accuracy 0.983333 | same corpus |
| Persisted context head | balanced accuracy 1.000000 | same corpus |
| 32 label placebos | mean 0.510328; max 0.748026 | same features and evaluation protocol |
| causal-softmax ablation | balanced accuracy 1.000000 | same router role, different attention normalizer |

The context signal is real. This corpus gives CRSA and causal softmax the same
balanced accuracy. The supported result is the router and its role program;
the two normalizers remain tied on this corpus.

Reproduce the report:

```bash
PYTHONPATH=src python scripts/crsa_route_eval.py
```

## Router-v2 structural scan

The placebo-controlled gate scan locates question structure in middle and
late-middle layers of Qwen3.8-27B. Layer 0 is dead; layers 18, 27, 36, 45,
and 54 exceed their placebo controls. The follow-on static value sketch scores
24% against a 32% placebo at 138.4 KB per question and is rejected.

Evidence:

- [`results/router_v2_stage1.json`](../results/router_v2_stage1.json)
- [`results/router_v2_stage2.json`](../results/router_v2_stage2.json)

## DeepSeek transport: remote network

These tests use the pinned official checkpoint over network range requests.
They measure transport envelopes only. All compared outputs are bit-identical.

| Experiment | Result | Scope |
|---|---:|---|
| one-ahead exact prefetch | 1.0928× mean speedup | ten routed-expert transport trials; cache-resident source | [`results/deepseek-v4-exact-prefetch-smoke.json`](../results/deepseek-v4-exact-prefetch-smoke.json) |
| window prefetch, cold network | 0.7330× mean speedup | four trials; exact expert ranges; candidate wins 3/4 but one network tail dominates the mean | [`results/deepseek-v4-exact-prefetch-window-network-smoke.json`](../results/deepseek-v4-exact-prefetch-window-network-smoke.json) |
| window prefetch, warm cache | 1.8311× mean speedup | ten trials; candidate wins 10/10 | [`results/deepseek-v4-exact-prefetch-window-smoke.json`](../results/deepseek-v4-exact-prefetch-window-smoke.json) |
| adjacent pair envelopes, cold network | 1.0763× mean speedup | ten trials; same bytes, fewer envelopes; candidate wins 7/10 | [`results/deepseek-v4-adjacent-range-network-smoke.json`](../results/deepseek-v4-adjacent-range-network-smoke.json) |
| adjacent pair envelopes, warm cache | 0.9822× mean speedup | ten trials; candidate wins 5/10 | [`results/deepseek-v4-adjacent-range-warm-smoke.json`](../results/deepseek-v4-adjacent-range-warm-smoke.json) |
| LM-head range batching | 1.4458× paired median; 4/4 wins | exact official LM head; 127 logical blocks; same 1,059,061,760 bytes | [`results/deepseek-v4-head-range-network-smoke.json`](../results/deepseek-v4-head-range-network-smoke.json) |

Network variance is visible in every report. The causalized local bundle
removes remote checkpoint transport from the production architecture.

## Local causal execution status

The local causal path now executes end to end. The repository contains and
tests:

- local positional Safetensors reads;
- append-only LiveCausal segments and crash recovery;
- checkpoint-bound causal weight plans;
- concurrent graph append with reader refresh;
- exact expert payload reads through graph routes;
- pager attachment for causal range resolution;
- complete stateful Qwen3.8 prefill, decode, and exact K=1–4 continuation;
- causalized Qwen3.5-0.8B transactional K=2 and K=4 draft providers;
- deterministic interval/RSS-pressure cyclic-GC control;
- native Prefix-Sinkhorn target verification with target-only commits;
- token-row Markov prediction, held-out prompt splitting, \(k\)-sweeps, and
  label-preserving placebo construction.

The local Qwen rows above are the current exact speed boundary. Their
receipts bind one continuation pair or one fixed prompt; broader throughput,
acceptance, and quality claims begin only after a frozen multi-prompt cohort.

## Reproduction gate

```bash
PYTHONPATH=src python -W error::ResourceWarning -m unittest discover -s tests
PYTHONPATH=src python -m compileall -q src scripts
git diff --check
```
