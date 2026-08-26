# Architecture

IMMER is a local runtime built from multiple execution planes. It composes
immutable neural weights, an append-only causal control plane, deterministic capability
organs, grounding, and persistent state behind explicit contracts.

## System map

```text
request -> CompositionRoot -> FERTIG exact first refusal
                              |
                              v
                         warm OoE hook
                    / verified       \ miss / novelty
                   v                  v
       Markov action executor     local causal Qwen3.8
       Crystal / Battery / Organ      |
                   |                  v
                   +----------> final FERTIG adjudication

continuous learning plane:

living frontier -> Qwen probe -> exact WeightCoordinate + projected pre/post hidden states
     -> SemanticWeightAtlas -> O1 signal -> contextual operator harvester
     -> whole / Attention core+residual / MLP core+residual families
     -> affine / permutation / Markov Crystals -> Birkhoff atom bases
     -> prompt-preserving sequence predictor -> O1 novelty/error evidence
     -> persistent algebra catalog -> Thompson/MAP-Elites selection -> Crystal VM
     -> replicated Markov agents -> world model / options / operator graph
     -> demand scheduler -> charged prefix + live residual suffix
     -> verifier outcome -> algebra and demand feedback
     -> opaque consequence word -> context-aware executable language
     -> self-hosted definition DAG -> compiled Crystal / bounded program
```

## 1. Immutable data plane

The weight plane contains original Safetensors. IMMER reads headers, tensors,
rows, and expert payloads by exact half-open byte range. Local reads use
positional I/O; remote reads exist for acquisition and controlled experiments.
Production execution targets local storage.

Every range stays bound to the checkpoint repository, immutable revision, and
layout fingerprint. A route built for one layout cannot silently address
another.

## 2. LiveCausal control plane

LiveCausal stores exact trigger-to-outcome relations in content-addressed,
append-only segments. Its manifest is hash-chained and crash-recoverable.
Queries load only the required graph neighborhood; no eager transitive closure
is built at mount time.

Within a causalized model bundle, LiveCausal maps semantic model coordinates
to exact range plans. Appending knowledge changes the wiring, not the weights.
The same local bundle therefore supports a stable tensor body and a growing
address graph.

## 3. Primary causal Qwen runtime

`src/immer/runtimes/qwen3_8/` implements the primary local neural path:

- authenticated causal-bundle mounting and exact range paging;
- stateful full-attention and Gated DeltaNet execution;
- native Causal Prefix Sinkhorn Attention;
- exact K=1–4 continuation and Qwen3.5 drafting;
- semantic state snapshots and prefix batteries;
- contextual cartography receipts tied to exact weight coordinates.

The anchor cache can restore the deepest exact prompt prefix before generation.
An exact hit uses the authenticated final-hidden seed; a shorter hit evaluates
only the suffix. The output token loop remains the ordinary Qwen loop.

## 4. Organism of Experts

OoE learns runtime actions around Qwen without modifying Qwen's weights or
internal router. Each stationary site is one exact execution identity:

```text
model pin + weight coordinate + weight graph revision
+ feature schema + action schema
```

O1/Atlas measurements provide contextual numeric sketches and verifier-bound
teacher transitions. Replicated agents retain action-conditioned sufficient
statistics. Executed PS-Lifted push-sum fuses those statistics into a finite
world model `P(s' | s, a)`. The planner builds previously unseen multi-step
solutions from one-step evidence and abstains when coverage, entropy, or peak
probability falls outside its authenticated gates.

Repeated verified trajectories become hierarchical options. Exact option
composition reduces primitive depth without adding multi-step teacher labels.
Balanced Sinkhorn assignment keeps option discovery one-to-one, and the
resulting option catalog remains receipt-bound. Structured transition tensors
can be stored as tensor-train/MPO factors; unsupported rank or error budgets
select the exact dense representation.

A mobile Markov token carries fading reservoir state and a Möbius rapidity
ledger. The continuous PS-Lifted reservoir retains delayed state across raw
fragments, fuses ridge sufficient statistics, and survives topology changes.
Hopfield energy supplies the calibrated novelty gate. Regime receipts make
nonstationary retention changes explicit rather than silently rewriting the
learned world.

The same agents now learn operator demand. UCB1 explores every unseen
materialized route before exploitation; verified negative evidence blocks the
exact failed route/revision/ABI until a later verified success replaces it.
PPM uses selection order rather than asynchronous verifier-completion order.
Explicit episodes build co-occurrence prefetch, and Ricci retention ranks
resident routes by `|R| exp(-alpha*logical_age)` while pins remain protected.

The action alphabet is `restore_anchor`, `execute_fertig`, `mount_organ`,
`probe_coordinate`, and `qwen_fallback`. Coverage is explicit. A partial
Crystal executes covered sources and abstains everywhere else.

Warm results are transactions. Executor-local verification creates a pending
decision; only final FERTIG adjudication commits saved Qwen forwards. Rejection
records zero savings. Crystal corruption is a hard integrity error. The local
causal Qwen bundle is the full neural weight substrate, one source of
cartography evidence, and an execution rail for unresolved neural work. The
stored-compute system has no mandatory teacher hierarchy: an admitted Crystal,
exact organ, continuation state, materialized route, or Qwen path executes
according to the selected contract.

The executable language sits above that action frontier. The sender sees the
intended action and emits an opaque word. The receiver sees only that word and
an authenticated context, chooses an action, and receives a one-shot verifier
consequence. Shared word/action evidence carries stable meaning across states;
context residual tables override it when consequences differ by state. Visit,
value, and margin gates reject valid but ungrounded or ambiguous words.

Factorized word slots learn from one whole-action consequence and recombine
held-out Cartesian action tuples. A frozen language snapshot maps admitted
words to the exact frontier artifacts they name: Crystals, programs,
materialized routes, options, organs, causal sites, residual plans, or Qwen
paths. The frontier binds the schemas and authorities used by that specific
action family; model-specific pins are present only for model-specific actions.

Production feedback enters through a joined bridge:

```text
opaque word + context
  -> ReceiverDecision
  -> exact ActionFrontier binding
  -> DemandRoutedExecutor / algebra verifier
  -> persistently committed source outcome
  -> hashed reward policy
  -> atomic receiver + sender update
  -> canonical language state CAS
```

Raw action labels and target values never enter receiver feedback. Demand
training accepts only the full joined execution receipt after route selection,
residual execution, external verification, outcome creation, and scheduler
settlement all agree.

Successful multi-step episodes are ordered by persisted selection time rather
than verifier completion time. Promotion also requires the declared step count
and a frontier-authorized terminal verifier. Repeated globally stable word
programs become definitions, resolve to local ComputePrograms, compile, and
enter the next action frontier under one promotion/migration receipt.

Frontier migration retains a Q row only for an unchanged action binding under
the same reward policy and a typed authority-transition proof. Route proofs
carry the complete graph-state payload chain and replay every append. Context
tables have a separate schema-identity gate. New actions receive deterministic
unused words and remain non-executable until consequence evidence clears the
normal visit/value/margin gates.

## 5. Stored compute and exact result cells

IMMER stores completed computation at two different abstraction levels.

`ComputeCrystal` is the general mechanism. It is a canonical typed program over
numerical values: affine, permutation, lookup, row-stochastic Markov, and
left-acting causal-mix operators share one bounded VM. Compatible homogeneous
chains fuse exactly. A `ComputeOperatorGraph` connects
verified one-step state transitions, plans a compatible route, and charges that
route under an authenticated charge basis. Charging fuses the route into a new
operator. Discharge applies the stored operator to values that did not exist
when it was charged and records equivalent source work, live work, and released
historical work. The bank and graph use content addresses, atomic manifests,
generation checks, and crash recovery.

Executable words reuse this substrate without flattening their definition
graph. A definition stores direct word references under the frozen language,
frontier, and named authority hashes. Missing dependencies repair recursively;
conflict, cycle, stale authority, and primitive redefinition fail closed. The
lexicon persists immutable histories and commits behind a CAS head.

For a homogeneous word, the compiler resolves each distinct child once and
fuses direct child Crystals bottom-up. Duplicate child hashes remain duplicate
execution positions. A reserved work-provenance extension carries the complete
transitive source work into the next level, so the root charge reports the
entire computation it replaces. Mixed operator families remain an ordered
bounded `ComputeProgram`; they never receive constant-discharge credit.

Residual discharge extends full-route charging. A future plan selects the
deepest compatible route that carries an authenticated charge basis, executes
that stored prefix, and applies only the remaining primitive suffix. A
materialized address without a charge basis receives no historical-work
credit. One receipt binds the
original plan, graph head, prefix route, both VM executions, tensor hashes,
provenance, equivalent work, live work, and released work.

Doubly-stochastic operators use constructive Birkhoff decomposition:

```text
measured kernel -> positive-support perfect matchings -> weighted atoms
                -> executable permutation Crystals + Markov Crystal
```

The bank stores `O(Kn)` permutation images, never a factorial basis. The
component count obeys `(n-1)^2+1`; weighted atom execution is checked against
the Markov Crystal on future inputs. Behavioral MAP-Elites retains diverse
Pareto operators, contextual Thompson sampling learns mutation families, and
Fiedler-projector distance proposes missing graph bridges without eigenvector
sign or basis ambiguity.

Demand-routed execution closes the learning loop:

```text
PPM context -> persisted UCB choice -> forced charged prefix
            -> live residual suffix -> external verifier
            -> positive/negative outcome -> next demand decision
```

The joined receipt binds every arrow and its scheduler state transition.
Operational exceptions create neutral abort events that remove unfinished
pulls; they never become fabricated quality failures. Verified success reward
is `1 + historical_work_released / equivalent_source_work`, so equal-quality
routes compete on computation returned from the past.

The exact affine-monoid plane generalizes stored operators beyond float64
matrices. An action carries an exact pair `(A,b)` over integer or modular state,
plus declarative phase and domain guards. Guarded fusion computes
`(A₂A₁, A₂b₁+b₂)` while retaining every intermediate guard stage. One runtime
therefore covers:

- explicit-depth LIFO stacks with overflow and underflow states;
- vector counters plus DFA phase for genuine `aⁿbⁿcⁿ`;
- length-, separator-, and phase-bound rolling fingerprints;
- strict signed Decimal-Horner;
- additive, multiplicative, and cyclic group-map actions.

Programs, initial/final states, and all execution receipts persist as one
replay-verified atomic bundle. The contextual algebra router selects among
verified programs with Thompson posteriors and keeps a Pareto MAP-Elites
catalog. Different state schemas compose as independent parallel lanes under a
joined ensemble receipt; the runtime rejects every attempt to reinterpret them
as one sequential ABI.

Algebraic Crystals learn the coordinate system of an operator family rather
than memorizing its outputs. Admission compares additive, multiplicative, and
finite-cyclic invariants under explicit score and margin gates. An exact group
accumulator snaps new measurements into the admitted group and executes long
compositions without replaying every primitive observation.

`ResultCell` is the sealed endpoint-specific class. It stores one complete cold
Qwen/FERTIG result under the exact model pin, code revision, tokenizer,
question hash, rendered-prompt hash, token-stream hash, system-prompt hash,
generation policy, feature receipt, verifier/evidence tuple, and final
judgment. Its warm executor
performs zero Qwen forwards and releases savings only after exact output,
FERTIG-status, and frozen-evaluator parity. A ResultCell cannot substitute for
a generic ComputeCrystal; its narrow binding is the reason it can return a
complete cached result safely.

The first real temporal holdout proves this endpoint: an authenticated
five-forward cold baseline executes warm with zero Qwen forwards while the raw
Qwen document and final semantic core remain exact. The frozen evaluator opens
once and verifies the output. FERTIG abstains on both paths, so the result is
evaluator-verified exact replay and its FERTIG semantic-certificate flag is
false.

## 6. O1 cartography and SemanticWeightAtlas

O1-State continuously measures surprise and learning progress over authentic
Qwen probe outcomes. The Atlas stores each immutable `MeasurementReceipt`
under its prompt, coordinate, intervention, model, weight-rail revision, and
append-only Atlas head.

The initial cartography manifest stays immutable while a separate hash-chained
frontier journal appends prompts, layers, weight sites, and interventions. Each
event expands the full prompt × job product. The scheduler atomically adopts
new cells without losing outcomes, histories, replay, stream state, or Atlas
coverage. `frontier-status` authenticates this state without opening model
weights; the durable idle runner waits for later events and records every run,
wait, failure, probe call, and harvested observation across restarts.
Attention coordinates follow the authenticated hybrid layer topology:
DeltaNet layers use `linear_attn.in_proj_qkv`, while full-attention layers use
`self_attn.q_proj`.

Atlas revisions expose exact hash-chain membership. Rollback, fork, forged
historical events, and a head change during proof construction fail closed.
Scheduler observation order can traverse authenticated historical revisions in
any order. Restore authenticates every seen revision, requires the initial
revision to be a history member, and requires the current revision to equal the
maximum seen head; teacher transitions retain their independent temporal order.
Crystal promotion uses a sealed two-phase transaction, so a crash during a
multi-site publication batch resumes from prepared state without repeating a
Qwen probe.

The probe also returns the deterministic hidden projections it already used
for evidence. Each float64 pre/post array is read-only and its byte hash must
match the sealed layer record. The live cartography script converts those
arrays into contextual transition receipts immediately after Atlas append and
feeds a separate persistent ComputeCrystal bank. Exact per-prompt runtime
receipts remain audit evidence; a stable runtime-family identity pools
compatible prompts and coordinates under one layer/action family. Intervention
mode and Qwen action schema prevent incompatible arms from sharing a fit.

The harvester fits earlier cursor-ordered observations and reserves the newest
one as holdout. It promotes affine, exact permutation, and stochastic Markov
operators only after train and holdout execution match. Numeric summaries and
placebo effects remain scalars; the runtime never reconstructs a hidden tensor
from summary statistics. Weight-site measurements sharing one prompt remain
separate Atlas records but count once inside an operator family, so fit and
holdout evidence is prompt-diverse rather than coordinate-duplicated.

The probe also captures five internal boundaries during the same layer
execution: normalized Attention input, Attention output, Attention residual,
normalized MLP input, and down-projected MLP output. These form Attention
core/residual and MLP core/residual transition families beside the whole-layer
pair. Stage filtering occurs before cloning; unused gate/up/activation tensors
never enter the cartography payload.

Signed projected sequences retain their prompt boundaries in a separate
predictive plane. A fixed seeded multi-timescale recurrence constructs
features per prompt, a ridge readout predicts the layer residual, one later
prompt selects feature/ridge configuration, and a separate final prompt yields
the holdout receipt. Raw X/Y content hashes block relabeled duplication. This
artifact's authority is restricted to error, novelty, and experiment-selection
evidence; executable replacement remains behind the exact Crystal verifier.

Every promoted family crosses directly into the executable algebra catalog.
The bridge rebinds the promotion to the current graph head, exact edge,
published Crystal, fit/holdout evidence, and a family-specific runtime verifier;
then it publishes a tagged `ComputeProgram`. Profile, bridge, candidate index,
and admission-time router head persist before router CAS and are audited on
restart. A selected program executes through `ComputeCrystalVM`; its joined VM
and verifier receipt becomes exact Thompson feedback. Parallel edges remain
individually addressable through `plan_exact_path()` and
`discharge_exact(route_sha256, value)`.

Realized causal sequence kernels use a distinct executable algebra.
`CAUSAL_MIX_FLOAT64` applies `K·V` on the final sequence axis, validates exact
zero future mass and row normalization, fuses kernels in execution order, and
charges under a causal-mix-specific verifier. Its left-action value semantics
remain separate from row-vector Markov probability transport.

## 7. DeepSeek-V4-Flash transport laboratory

`src/immer/runtimes/deepseek_v4/` implements the checkpoint math directly:

- checkpoint config and pinned provenance;
- BF16/FP8/FP4 decoding;
- MLA attention and routed/shared MoE execution;
- exact expert paging;
- layer-major scoring and authenticated resume;
- stateful autoregressive generation;
- FERTIG draft verification;
- causal weight resolution;
- label-free route learning and placebo evaluation.

DeepSeek remains the transport and control laboratory. Its local/remote split
rails, exact range accounting, causal append machinery, state transport, and
placebo discipline feed the primary Qwen architecture. It is not the default
teacher path.

## 8. Markov transport controller

DeepSeek's official router emits selected expert IDs for every active token
row. IMMER records those rows without labels and estimates layer-to-layer
transition distributions.

For source layer \(\ell\), current expert \(i\), and candidate expert \(j\):

\[
\hat P_\ell(j\mid i)=
\frac{N_\ell(i,j)}{\sum_k N_\ell(i,k)}.
\]

Token-row mixtures are aggregated into a full next-layer distribution. The
runtime can evaluate any \(k\), compare against target-layer marginals and a
label-preserving placebo, and convert ranked candidates into exact range
plans. Official router output remains authoritative.

## 9. Exact execution plane

The `ExactCascade` owns exact arithmetic routing:

1. the frozen S3 host selects among four SHA-addressed organs;
2. FERTIG verifies the result or handles a grounded fallback;
3. contradictions and unsupported structures return abstention.

No online learning step can mutate this path. This makes exact results usable
as certificates around neural inference.

## 10. Causal Prefix Sinkhorn Attention

CRSA is IMMER's Causal Prefix Sinkhorn Attention mechanism. It assigns heads
explicit causal roles:

- **Local:** bounded recent context;
- **Balanced:** prefix-mass-balanced Sinkhorn attention;
- **Free:** ordinary causal softmax.

The deployed role-complete program is two Local heads, one Balanced head, and
one Free head. The Free head preserves unrestricted causal reach and is
bit-exact with causal softmax for the same logits.

## 11. Persistent state

The O(1)-state runtime maintains a separate life stream with surprise-gated
updates, replay, and sleep consolidation. It can learn without rewriting the
frozen exact host or the immutable initial frontier. Frontier events extend its
measurement work while preserving the complete prior learning stream.

## Invariants

- Weight bytes are immutable.
- Graph appends are durable and independently verifiable.
- Atlas history proves exact sequence/event membership.
- Remote model sources require immutable revisions.
- Every range read is identity-bound and byte-accounted.
- Transport hints never change official router decisions.
- OoE executes only covered, calibrated, verifier-bound Crystal actions.
- ComputeChargeReceipt binds the source program, parent operators, fused
  operator, exact fusion check, and verifier receipt; the materialized charged
  route adds the durable bank publication and graph revision.
- Generic ComputeCrystals accept unseen runtime values; exact ResultCells accept
  only their complete sealed cold binding.
- Saved Qwen forwards commit only after final result verification.
- Exact capability execution either returns a verified answer or abstains.
- Private state and operational topology stay outside the public repository.
