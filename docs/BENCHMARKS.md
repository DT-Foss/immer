# Benchmarks

Release: Unreleased · 2026-08-27

Integration base for the new FERTIG-action and predictive-quotient rows:
`45778a4aeac55a8f1a7bcb47df73fee4972e1228`.

Final warning-fatal regression boundary for this cut: `1,814/1,814`, `OK`,
`591.551 s`.

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
| Live FERTIG action learning | `16/16` warm E2E; `8/8` O1 sites; answers `81`, `310`; `0` Qwen forwards | each transaction joins one real Atlas measurement, authenticated Qwen/O1 feature, exact FERTIG certificate, independent replay verifier, append-only trace, verified teacher transition, and persistent controller update; no raw question enters the receipt | [`execution_learning.py`](../src/immer/runtimes/ooe/execution_learning.py); [`fertig_executor.py`](../src/immer/runtimes/ooe/fertig_executor.py); [`ooe_fertig_action_learning.py`](../scripts/ooe_fertig_action_learning.py); private report body `b6b9b3870a1ac8c79463906e0d421992b9ca0a3b3cbce11ae79d22b8bbb6e598`, file `d536b54950dfb1027c610953827d0d210f3212d6a40f84897f6e37055f3466e6` |
| DeepSeek V4 MMLU | 3/4 correct; 43/43 layers complete | exact `off` mode on one fixed four-item high-school-geography slice | [`results/deepseek-v4-mmlu-off-4-exact-v4-window2x3-receipt.json`](../results/deepseek-v4-mmlu-off-4-exact-v4-window2x3-receipt.json) |
| DeepSeek V4 + FERTIG | 6/8 correct; 0 wrong; 2 abstentions | fixed eight-item GSM8K failure slice; five exact certificates plus one model-verified answer | [`results/deepseek-v4-fertig-fusion.json`](../results/deepseek-v4-fertig-fusion.json) |
| Qwen3.8 + FERTIG | 5/8 exact; 0 wrong; 3 quarantined | fixed eight-item GSM8K failure slice; model agreement cannot answer without an independent certificate | [`results/qwen38_fertig_fusion.json`](../results/qwen38_fertig_fusion.json) |

The four-item and eight-item rows are integration slices. Their claims stop at
those exact item sets.

## Markov-OoE and compute batteries

| Benchmark | Result | Exact scope | Evidence boundary |
|---|---:|---|---|
| Standalone Markov-OoE | local `17.28%`; reversible `49.84%`; PS-Lifted `96.10%`; warm `100%`; shuffled Crystal `13.50%` | six finite operator families; local noisy one-step teacher transitions; 250 unseen plans and 5,000 longer unseen compositions | imported package manifest `28/28`; production port and regression tests in [`runtimes/ooe`](../src/immer/runtimes/ooe) |
| Consequence-grounded executable language | primitive `100%`; unseen programs `100%`; context-dependent word `100%`; held-out grammar `100%`; option words `100%`; cultural child `100%`; self-hosted macro `100%` | five independent seeds × 1,000 held-out programs; receiver sees word + context + chosen-action consequence only; shuffled semantics `2.02%`, fixed no-message policy `0.98%`, no-action abstain `2.90%`, shuffled grammar `5.00%`, holistic held-out `0%`; every unassigned in-vocabulary word abstains | [`markov_language.py`](../src/immer/runtimes/ooe/markov_language.py); [`ooe_markov_language_benchmark.py`](../scripts/ooe_markov_language_benchmark.py); [`test_ooe_markov_language.py`](../tests/test_ooe_markov_language.py); [`test_ooe_markov_language_benchmark.py`](../tests/test_ooe_markov_language_benchmark.py) |
| Recursive executable word DAG | `8,191` actions → `36` references → one charged Crystal; exact parity; `8.380x` authenticated-VM speedup | depth 12, five seeds, 25 future states per seed; flat VM `0.327760 s`, compiled VM `0.039530 s`, one-time compile `0.784953 s`; `99.5605%` knowledge-reference and `99.9878%` deployment-symbol reduction; `1,638,000` historical vs. `200` live work units | [`executable_lexicon.py`](../src/immer/runtimes/ooe/executable_lexicon.py); [`compute_crystals.py`](../src/immer/runtimes/ooe/compute_crystals.py); [`test_ooe_executable_lexicon.py`](../tests/test_ooe_executable_lexicon.py) |
| Production language outcome bridge | exact Demand route/outcome/state binding; positive, negative, abort, stale revision, CAS rollback, replay, foreign graph, and forged authority proof covered | receiver-selected materialized route executes through `DemandRoutedExecutor`; language feedback accepts only the joined receipt after persistent Demand settlement; graph-growth proof embeds and replays every canonical graph state | [`language_bridge.py`](../src/immer/runtimes/ooe/language_bridge.py); [`test_ooe_language_bridge.py`](../tests/test_ooe_language_bridge.py) |
| Live O1 controller export | `40/40` probes; `200` authenticated observations; `8/8` promoted site kernels exported exactly | official local Qwen frontier; every `CrystalPayload` is rederived from controller snapshot/coverage, converted to `ComputeCrystal.markov`, checked on the identity basis, and sealed under one eight-action frontier plus append-only bank anchor | [`controller_crystal_bridge.py`](../src/immer/runtimes/ooe/controller_crystal_bridge.py); [`test_ooe_controller_crystal_bridge.py`](../tests/test_ooe_controller_crystal_bridge.py) |
| FERTIG-enriched controller frontier | `8/8` exported policies choose `execute_fertig` from the `execute_fertig` row | the execution-learned controller is cloned, every site is re-promoted, all exact 5x5 policies export to `ComputeCrystal.markov`, and restore rechecks the source snapshot, manifest, coverage, Atlas/model pins, kernels, sequential bank publications, and final frontier | [`ooe_controller_action_frontier.py`](../scripts/ooe_controller_action_frontier.py); frontier `44074166d5b7db8b39b43100b95aeffa48e4c80bd114e13f8079150e77315e41`; export `1663d7524460b9677cde7a5bc92218e2c2429d489964c8bd1a444763fbee590d`; report body `550d707d5093ab3c464bac4d94c068f978430c92def59364b20aca3eb4903715` |
| Learned predictive-state quotient | positive `4→2`; non-bisimilar `4→4`; held-out TV `0`; disturbed TV `1`; reversed-order bytes identical | exact rational empirical future laws; equivalence only when conditional future distributions match; canonical quotient state is independent of observation order | [`predictive_quotient.py`](../src/immer/runtimes/ooe/predictive_quotient.py); [`test_ooe_predictive_quotient.py`](../tests/test_ooe_predictive_quotient.py) |
| Live FERTIG predictive quotient | site-local `9→1`, `9x`; global trace-topology control `9→9`; refinement rounds `1`; terminal `0`; censored `0` | all 16 live FERTIG learning receipts; controller state is `(site, source-action)` and its successor is `(same site, executed action)`; the control uses the global trace successor on identical evidence | [`ooe_predictive_quotient.py`](../scripts/ooe_predictive_quotient.py); quotient `064dd640c673a3e86dca24513663a1c3a7d2793ed279d2b0b9d2667081e15ae1`; private file SHA `44e3853cc9bb1953feb01146f4b634a75247cffdc5e129b80e6d7df7c5eaa3fa` |
| Exact live router blanket | `8→2` labels per feature; `128→32` centroid distances over 16 features; `75%` less distance work; exact decision parity | each closure is published only after a full scan proves that it contains the input site, global nearest and runner-up; receipt is exact-feature and router-calibration bound, with no unseen-feature generalization | [`predictive_blanket.py`](../src/immer/runtimes/ooe/predictive_blanket.py); [`ooe_router_blanket.py`](../scripts/ooe_router_blanket.py); report `d27cc87e943cbf08512f9863d95964e1e5f8702557145f9659ee9c0e3cac18dd` |
| Qwen boundary blanket | fixed residual identity `2/6`; calibration NRMSE `0.0016464655`; unseen-L54 `0.0016575120` vs Full `0.0017912050`; equal-size rank `1/16`; LOQO `5/5` | 40 atomic prompt×layer clusters, 200 authenticated transition arrays, 4,096 aligned rows; selection uses train+calibration only, topology holdout is later and layer-unseen, prompt LOQO is a separate audit; fit, holdout, LOQO folds, and the top-level report are independently recomputed | [`boundary_blanket.py`](../src/immer/runtimes/ooe/boundary_blanket.py); [`ooe_boundary_blanket.py`](../scripts/ooe_boundary_blanket.py); live file `ed30426134d0d412cac74c3d48fa35fe74c476d2e3ae51a033a41049fed990eb` |
| Exact Demand lag blanket | lags `(1,3)`; Selected/Full `9/10`; contiguous PPM `4/5`; Random `1/2`; Marginal `2/5`; `144/336` historical work units released | 30 verified successful explicit outcomes in three chronological episodes; full report replay rebuilds corpus, fit, validation, prediction, four routed executions, and exact tensor outputs; runtime order is Blanket → PPM → complete UCB with incompatible-route guard | [`demand_blanket.py`](../src/immer/runtimes/ooe/demand_blanket.py); [`ooe_demand_blanket_benchmark.py`](../scripts/ooe_demand_blanket_benchmark.py); report file `021d54d70c59500bca42e65a20fb81d27241f36c9814eb251cf9176e227d6038` |
| Prefix-Sinkhorn operator transport | per-head held-out residual `1.33e-17`; global `9.57e-4`; identity `1.06e-3`; random `5.78e-2` | deterministic causal Prefix-Sinkhorn mechanism fixture; prompt/position groups are atomic and strictly temporal; rank, identification gap, system condition, and `cond(C)` gate every fit; current Qwen Atlas correctly reports the pre-blend per-head operator as missing | [`operator_transport.py`](../src/immer/runtimes/ooe/operator_transport.py); fit `e6d8ac8ddd3e17a8e45442f5eabd9cf9c8f71fac7b99238198879ac570374916`; holdout `8631a2577f041e677ba966e4be3a6364187f459214ec8662e8e4c6ec9ccadaaf` |
| Joint gate/up collision boundary | `k=1` wrong collisions train/cal/holdout `12/4/8`, blocked; `k=2` calibration `4/4` and holdout `8/8`, zero wrong, `112` bytes; `k=4` zero wrong, `240` bytes | synthetic exact mechanism control over 22 candidate/control models; bases and quantizer scales are train-only; only candidate families with zero prior/current wrong collisions and a verified hit can promote; real Harvester emits request `70f0e455…` and no live fit | [`subspace_battery.py`](../src/immer/runtimes/ooe/subspace_battery.py); fit `35ac9b73b8300549e15ac84dcc291efdb28bcaf9daae87b36c5055043df97a43`; holdout `e950026772b703ed43efd3eecee51705cb8c85f7d0f96bb4e69d76c453af9488` |
| Native Prefix-Sinkhorn sidecar | first 32 absolute positions; bounded capture peak `49,664 B`; disabled/alpha-zero paths emit zero operator rows; matching sidecar reuses with zero probe | exact pre-blend `streaming_prefix_log.routed`, four native heads, tokenwise positions and causal support; O1 tests cover missing/matching/wrong sidecars, raw absence, Atlas-to-sidecar crash repair, and dirty-orphan status | [`native_crsa.py`](../src/immer/runtimes/qwen3_8/native_crsa.py); [`cartography_probe.py`](../src/immer/runtimes/qwen3_8/cartography_probe.py); [`qwen38_o1_cartography.py`](../scripts/qwen38_o1_cartography.py) |
| Exact MLP evidence bank | canonical capture plan `25/10/5`; fixture→live replay creates all `40` entries in both modes (`80` receipts); orphan objects/histories force `audit_clean=false` | BF16 bit-exact binary framing, no-replace objects, CAS journal, four crash-recovery boundaries, ordered MLP stages, projection-verifier protocol, fixture rejection from production corpus | [`qwen_mlp_evidence.py`](../src/immer/runtimes/ooe/qwen_mlp_evidence.py); [`qwen38_o1_mlp_evidence.py`](../scripts/qwen38_o1_mlp_evidence.py) |
| First live executable controller language | frozen accuracy `100%`; unseen greedy choices `1,000/1,000`; reusable bootstrap holdout `12/12` | eight real promoted site-policy actions; mounted one-shot outcome executor recomputes intended/selected operators and gives `+1/-1` only after exact action, artifact, output, source-extension, frontier, and bank-authority verification; three held-out clones start at the frozen learner and never update it | [`language_bridge.py`](../src/immer/runtimes/ooe/language_bridge.py); [`controller_language_intelligence.py`](../src/immer/runtimes/ooe/controller_language_intelligence.py); private sealed reports `ce5515d…` and `a2311ccf…` |
| First live compiled controller word | four actions → one charged Crystal; three support executions; `3,375` historical work units released; max delta `1.110223e-16` | word sequence is expressed in the frozen live dialect, resolved through the eight-action frontier, compiled bottom-up, persisted in the executable lexicon, and replayed on 25 future probability vectors | [`controller_language_intelligence.py`](../src/immer/runtimes/ooe/controller_language_intelligence.py); [`ooe_controller_language_bootstrap.py`](../scripts/ooe_controller_language_bootstrap.py); [`test_ooe_controller_language_intelligence.py`](../tests/test_ooe_controller_language_intelligence.py) |
| First live sibling-dialect macro transfer | different surface assignments `8/8`; semantic translation `8/8`; four localized words → one charged Crystal; `3,375` work units; max delta `2^-52` | two independently seeded languages share only the authenticated eight-action O1 frontier; three isolated controller-support chains authorize the PortableWordProgram by exact bridge hash, then the target snapshot supplies its own words and local compilation | controller-language portability bridge plus [`dialect_mesh.py`](../src/immer/runtimes/ooe/dialect_mesh.py); private live report `7b4a026d…` |
| Continual executable-language growth | base `100%`; retained old actions `100%`; promoted action learned after mean `328` episodes; final `100%`; no-memory `0%` | five seeds, 5,000 base + 5,000 growth episodes; compiled word promoted as fourth action; initial new-action abstention `5/5`; context-schema shift drops local evidence but global accuracy remains `100%`; reward-policy shift resets `4/4` and scores `0%` | [`language_intelligence.py`](../src/immer/runtimes/ooe/language_intelligence.py); [`ooe_language_growth_benchmark.py`](../scripts/ooe_language_growth_benchmark.py); [`test_ooe_language_intelligence.py`](../tests/test_ooe_language_intelligence.py) |
| Second-generation compute word | constant discharge `5/5`; historical work released `400 → 1,200` units | newly learned promoted macro becomes a primitive word in the next snapshot; a second definition uses it twice plus one original action and compiles bottom-up to one Crystal across 25 future inputs | same continual-growth harness and tests |
| Executable dialect mesh | verified translation `60,000/60,000`; worst-context accuracy `100%`; direct surface `6.84%` programs / `23.41%` tokens; permutation placebo `1.42%` / `19.58%` | five seeds × five dialects × three separately trained contexts; fully shared six-token inventory with `68.66%` semantic collisions; every ordered source/target/context receipt executes 200 unseen 2–8-action programs; all four target dialects have 3/3 distinct mappings and zero global action words | [`dialect_mesh.py`](../src/immer/runtimes/ooe/dialect_mesh.py); [`dialect_intelligence.py`](../src/immer/runtimes/ooe/dialect_intelligence.py); [`ooe_dialect_mesh_benchmark.py`](../scripts/ooe_dialect_mesh_benchmark.py); [`test_ooe_dialect_mesh.py`](../tests/test_ooe_dialect_mesh.py) |
| Portable semantic macro | target localization/execution `75/75`; changed binding/authority rejected `5/5`; mean `150` historical work units released | one source-discovered four-action program binds semantic actions, exact artifacts, discovery, definition, authorities, and support trajectories; every target uses context-local words and a context-bound lexicon state, then compiles locally on 25 future inputs | same dialect-mesh harness plus [`test_ooe_dialect_intelligence.py`](../tests/test_ooe_dialect_intelligence.py) |
| PS-Lifted consensus | `338 → 64` rounds (`5.28125x` fewer) | original fixed barbell topology and tolerance; raw six-kernel payload `588 B` | [`test_ooe_math_consensus.py`](../tests/test_ooe_math_consensus.py); [`test_ooe_qwen_bridge_controller.py`](../tests/test_ooe_qwen_bridge_controller.py) |
| Action-conditioned world model | PS-Lifted `250/250`; central table `250/250`; 12-replica local cohort `226/3,000 = 7.5333%`; shuffled-action placebo `89/250`; no-memory `68/250` | 17 states, four actions, 12 replicas, 68 authenticated one-step observations, 250 unseen 4–10-step tasks per policy evaluation, zero multi-step labels | [`intelligence.py`](../src/immer/runtimes/ooe/intelligence.py); [`ooe_intelligence_benchmark.py`](../scripts/ooe_intelligence_benchmark.py); [`test_ooe_intelligence_benchmark.py`](../tests/test_ooe_intelligence_benchmark.py) |
| Markov self-calibration and topology shift | fixed-`pc` barbell `96` rounds → self-calibrated barbell `58`; complete topology `28`; success `100%` throughout; maximum fused-count delta `6.02e-9` | same fixed-seed world-model trial; self-calibration changes the consensus envelope, then topology changes after evidence collection | [`consensus.py`](../src/immer/runtimes/ooe/consensus.py); [`test_ooe_math_consensus.py`](../tests/test_ooe_math_consensus.py) |
| Hierarchical options | primitive depth `8 → 1`; success probability `1.0` | seven discovered options from four verified trajectories; exact contraction ledger; no multi-step teacher labels | [`options.py`](../src/immer/runtimes/ooe/options.py); [`contraction_ledger.py`](../src/immer/runtimes/ooe/contraction_ledger.py); [`test_ooe_options_mpo.py`](../tests/test_ooe_options_mpo.py) |
| Continuous PS-Lifted reservoir | fused `90.8359%`; local mean `82.1626%`; no-memory `50.4532%`; shuffled-label placebo `54.2800%` | eight replicas, 32-node fixed reservoir, 500 training steps per replica, 1,000-step test, delayed-state target at lag seven | [`reservoir.py`](../src/immer/runtimes/ooe/reservoir.py); [`reservoir_intelligence.py`](../src/immer/runtimes/ooe/reservoir_intelligence.py); [`test_ooe_reservoir_intelligence.py`](../tests/test_ooe_reservoir_intelligence.py) |
| Generic ComputeCrystal discharge | four primitive operators → one live operator; `27,648` authenticated historical work units released; maximum delta `1.7764e-15` | route charged before 128 unseen six-dimensional inputs existed; exact reopen/replay; disconnected placebo abstains | [`compute_crystals.py`](../src/immer/runtimes/ooe/compute_crystals.py); [`compute_graph.py`](../src/immer/runtimes/ooe/compute_graph.py); [`test_ooe_crystal_intelligence.py`](../tests/test_ooe_crystal_intelligence.py) |
| CausalMix ComputeCrystal | exact zero future mass; eight-step maximum fused delta `1.3323e-15` | randomized causal row-normalized kernels through dimension 32; left action on future value channels; canonical persistence, VM, dedicated fusion verifier, route charge, exact-route discharge | [`compute_crystals.py`](../src/immer/runtimes/ooe/compute_crystals.py); [`compute_graph.py`](../src/immer/runtimes/ooe/compute_graph.py); [`test_ooe_compute_crystals.py`](../tests/test_ooe_compute_crystals.py); [`test_ooe_compute_graph.py`](../tests/test_ooe_compute_graph.py) |
| Charged-prefix residual discharge | two charged prefix steps + two live suffix steps; `9,216` historical work units released; maximum delta `1.3323e-15` | same 128 unseen six-dimensional inputs; prefix was charged before inputs existed; suffix remains primitive and live; uncharged materialized routes receive zero reuse credit | [`residual_execution.py`](../src/immer/runtimes/ooe/residual_execution.py); [`test_ooe_residual_execution.py`](../tests/test_ooe_residual_execution.py); [`test_ooe_crystal_intelligence.py`](../tests/test_ooe_crystal_intelligence.py) |
| Constructive Birkhoff Crystal basis | `33` atoms under exact bound `50`; maximum weighted-basis delta `6.6613e-16` | fixed seeded 8×8 doubly-stochastic operator; O(Kn) atom storage; 96 future inputs; full basis, receipt, reopen, inverse-permutation semantics, idempotence, and non-DS rejection tested | [`bvn_search.py`](../src/immer/runtimes/ooe/bvn_search.py); [`bvn_crystals.py`](../src/immer/runtimes/ooe/bvn_crystals.py); [`test_ooe_bvn_search.py`](../tests/test_ooe_bvn_search.py); [`test_ooe_bvn_crystals.py`](../tests/test_ooe_bvn_crystals.py) |
| Autonomous operator search | `12/64` MAP-Elites cells occupied; `14` Pareto elites; preferred mutation arm `32/32` late choices | 64 constructive DS candidates; three mutation families; fixed seed; Fiedler missing-bridge priority `0.409362` versus `0.120790` for the highest existing edge after direct-edge penalty | [`bvn_search.py`](../src/immer/runtimes/ooe/bvn_search.py); [`crystal_intelligence.py`](../src/immer/runtimes/ooe/crystal_intelligence.py); [`test_ooe_crystal_intelligence.py`](../tests/test_ooe_crystal_intelligence.py) |
| Verified operator demand | unseen-first UCB1; exact negative-edge blocking; selection-ordered PPM; episode-scoped co-occurrence; byte-exact Ricci retention | deterministic contract suite with asynchronous outcome settlement, standalone-event placebo, ABI/revision mismatch, forged semantic receipt, concurrent CAS, crash, rollback, fork, tamper, pins, and full program/Crystal/charge byte accounting | [`demand_scheduler.py`](../src/immer/runtimes/ooe/demand_scheduler.py); [`test_ooe_demand_scheduler.py`](../tests/test_ooe_demand_scheduler.py) |
| Demand-routed production loop | prefix rewards `1.25` and `1.50`; decision three selects the deeper prefix | same unseen 128-input affine route; exact chain PPM/UCB → forced prefix → residual VM → `1e-12` parity verifier → persisted outcome; operational verifier failure produces neutral abort and zero reward poisoning | [`demand_execution.py`](../src/immer/runtimes/ooe/demand_execution.py); [`test_ooe_demand_execution.py`](../tests/test_ooe_demand_execution.py); [`test_ooe_crystal_intelligence.py`](../tests/test_ooe_crystal_intelligence.py) |
| Exact guarded affine monoids | stack depth `53/64` exact and depth `65` dead; `aⁿbⁿcⁿ` `4/4`; placebos `7/7`; Decimal 128 digits exact | integer/per-coordinate modular `(A,b)` actions; guard-preserving fusion; explicit phase/dead/capacity/bit bounds; deliberate modular fingerprint collision becomes candidate and fails exact byte verification; one atomic replay-verified bundle | [`affine_monoid.py`](../src/immer/runtimes/ooe/affine_monoid.py); [`affine_intelligence.py`](../src/immer/runtimes/ooe/affine_intelligence.py); [`test_ooe_affine_monoid.py`](../tests/test_ooe_affine_monoid.py); [`test_ooe_affine_intelligence.py`](../tests/test_ooe_affine_intelligence.py) |
| Contextual algebra meta-agent | correct `120/120`; shuffled-context placebo `0/90`; Stack→Decimal recovery `30/30` | three verified Stack/Fingerprint/Decimal programs, contextual Thompson updates, three-cell MAP-Elites catalog, exact restart; handcrafted selections and stale verifier outcomes rejected | [`algebra_agents.py`](../src/immer/runtimes/ooe/algebra_agents.py); [`affine_intelligence.py`](../src/immer/runtimes/ooe/affine_intelligence.py); [`test_ooe_algebra_agents.py`](../tests/test_ooe_algebra_agents.py) |
| Prompt-preserving contextual sequence model | selected MSE `6.726089e-05`; tuned pointwise `5.803816e-02`; `862.881x` lower | fixed variable-length synthetic recurrence; train prompts + later validation selection + untouched holdout; token/output placebos `3.1959/3.1541`; raw-content dedup and 1 GiB default aggregate preflight | [`contextual_sequence.py`](../src/immer/runtimes/ooe/contextual_sequence.py); [`test_ooe_contextual_sequence.py`](../tests/test_ooe_contextual_sequence.py) |
| Harvested executable algebra routing | contextual `174/180`, late `60/60`, final `30/30`; shuffled-context placebo `49/180`, late `13/60`, final `5/30` | three held-out affine/permutation/Markov promotions converted to tagged `ComputeProgram` candidates; exact VM execution and verifier feedback on every decision; fixed seed `20260826` | [`harvest_algebra_bridge.py`](../src/immer/runtimes/ooe/harvest_algebra_bridge.py); [`harvest_algebra_intelligence.py`](../src/immer/runtimes/ooe/harvest_algebra_intelligence.py); [`ooe_harvest_algebra_benchmark.py`](../scripts/ooe_harvest_algebra_benchmark.py); [`test_ooe_harvest_algebra_intelligence.py`](../tests/test_ooe_harvest_algebra_intelligence.py) |
| Exact parallel-route choice | requested route SHA survives plan, charge, reopen, and discharge | two charged affine routes share the same source and goal but produce distinct outputs; `discharge_exact()` executes each requested route while canonical discharge retains cost-based selection | [`compute_graph.py`](../src/immer/runtimes/ooe/compute_graph.py); [`test_ooe_compute_graph.py`](../tests/test_ooe_compute_graph.py) |
| Heterogeneous exact ensemble | three independent ABIs, one joined receipt, replay exact | Decimal, Fingerprint, and Stack execute in parallel; each lane embeds program and initial state, replays standalone, and rejects swapped, missing, cross-program, same-schema, or verifier-artifact splices | [`algebra_agents.py`](../src/immer/runtimes/ooe/algebra_agents.py); [`test_ooe_algebra_agents.py`](../tests/test_ooe_algebra_agents.py) |
| Algebraic Crystals | additive `0.999999985`, multiplicative `0.999999844`, cyclic `0.999998261`; all length-12 results exact | nine support points per continuous family, 12 cyclic supports, unseen length-12 composition; permuted placebo score `0.155535` and rejected | [`algebraic_crystals.py`](../src/immer/runtimes/ooe/algebraic_crystals.py); [`test_ooe_algebraic_crystals.py`](../tests/test_ooe_algebraic_crystals.py) |
| Structured MPO kernel | `1,408 / 8,192` numeric bytes (`0.171875`); relative error `3.58e-16` | structured action-conditioned tensor; unstructured tensor selects authenticated exact-dense fallback when rank budget is exceeded | [`mpo.py`](../src/immer/runtimes/ooe/mpo.py); [`test_ooe_options_mpo.py`](../tests/test_ooe_options_mpo.py) |
| Hopfield novelty gate | `2/2` in-distribution accepted; `0/3` out-of-distribution false accepts | calibrated per-site energy and gap thresholds; removing the novelty gate accepts all three OOD cases | [`novelty.py`](../src/immer/runtimes/ooe/novelty.py); [`test_ooe_novelty.py`](../tests/test_ooe_novelty.py) |
| Qwen native anchor battery | `130.457495 s → 91.304167 s`; `1.428823x`; `30.012326%` demand latency removed | one official local CPU-BF16 Qwen3.8 cell; 65-token charged invariant and previously unknown 33-token suffix | private sealed receipt; [`anchor_battery.py`](../src/immer/runtimes/qwen3_8/anchor_battery.py); [`test_qwen3_8_adapter.py`](../tests/test_qwen3_8_adapter.py) |
| O1 → Atlas → OoE integration | live `10/10` jobs, `15` measurements, two sites, Atlas revision `15`, `501.95 s`; zero duplicate probes on Atlas reuse | five frozen public-GSM8K prompt identities; real contextual local-Qwen measurements at passive layer 0 and native Sinkhorn layer 27 | [`qwen38_o1_cartography.py`](../scripts/qwen38_o1_cartography.py); [`test_qwen38_o1_cartography.py`](../tests/test_qwen38_o1_cartography.py) |
| Living O1/Qwen frontier | combined cartography contracts `52/52`; exact append/resume/rollback/crash behavior | deterministic prompt × layer × site × intervention expansion; journal append leaves initial manifest immutable; scheduler preserves all O1/Atlas history; idle runner resumes and consumes later work; repeated sites from one prompt count once per operator family | [`qwen38_o1_cartography.py`](../scripts/qwen38_o1_cartography.py); [`cartographer.py`](../src/immer/runtimes/o1_state/cartographer.py); [`operator_harvester.py`](../src/immer/runtimes/ooe/operator_harvester.py); [`test_qwen38_o1_cartography.py`](../tests/test_qwen38_o1_cartography.py); [`test_o1_cartographer.py`](../tests/test_o1_cartographer.py); [`test_ooe_operator_harvester.py`](../tests/test_ooe_operator_harvester.py) |
| Real Qwen contextual operator stream | `40/40` projected transitions accepted into ten layer/sublayer families; four prompt contexts per family | four distinct prompts through the committed tiny causal-Qwen runtime; two measured layers × whole/Attention-core/Attention-residual/MLP-core/MLP-residual; exact per-prompt receipts remain distinct | [`cartography_probe.py`](../src/immer/runtimes/qwen3_8/cartography_probe.py); [`operator_harvester.py`](../src/immer/runtimes/ooe/operator_harvester.py); [`qwen38_o1_cartography.py`](../scripts/qwen38_o1_cartography.py); [`test_qwen3_8_cartography_probe.py`](../tests/test_qwen3_8_cartography_probe.py) |
| Qwen sublayer boundary stream | five operator families per measured layer | whole-layer, Attention core/residual, and MLP core/residual sketches captured inside the same forward; five stages persisted, arrays read-only/hash-bound, unused large MLP stages filtered before clone | [`model.py`](../src/immer/runtimes/qwen3_8/model.py); [`cartography_probe.py`](../src/immer/runtimes/qwen3_8/cartography_probe.py); [`operator_harvester.py`](../src/immer/runtimes/ooe/operator_harvester.py); [`test_qwen3_8_cartography_probe.py`](../tests/test_qwen3_8_cartography_probe.py) |
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
