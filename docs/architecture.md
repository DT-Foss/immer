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
       Crystal action executor    local causal Qwen3.8
       Battery / FERTIG / Organ       |
                   |                  v
                   +----------> final FERTIG adjudication

continuous learning plane:

Qwen probe -> exact WeightCoordinate -> SemanticWeightAtlas -> O1 signal
     -> replicated Markov site agents -> PS-Lifted consensus -> CrystalStore
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
teacher transitions. Replicated `MarkovPDAgent` kernels are fused by an
executed PS-Lifted push-sum, quantized, and published as immutable Crystals.
A mobile Markov token carries fading reservoir state and a Möbius rapidity
ledger; both participate in the execution gate.

The action alphabet is `restore_anchor`, `execute_fertig`, `mount_organ`,
`probe_coordinate`, and `qwen_fallback`. Coverage is explicit. A partial
Crystal executes covered sources and abstains everywhere else.

Warm results are transactions. Executor-local verification creates a pending
decision; only final FERTIG adjudication commits saved Qwen forwards. Rejection
records zero savings. Crystal corruption is a hard integrity error.

## 5. O1 cartography and SemanticWeightAtlas

O1-State continuously measures surprise and learning progress over authentic
Qwen probe outcomes. The Atlas stores each immutable `MeasurementReceipt`
under its prompt, coordinate, intervention, model, weight-rail revision, and
append-only Atlas head.

Atlas revisions expose exact hash-chain membership. Rollback, fork, forged
historical events, and a head change during proof construction fail closed.
Crystal promotion uses a sealed two-phase transaction, so a crash during a
multi-site publication batch resumes from prepared state without repeating a
Qwen probe.

## 6. DeepSeek-V4-Flash transport laboratory

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

## 7. Markov transport controller

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

## 8. Exact execution plane

The `ExactCascade` owns exact arithmetic routing:

1. the frozen S3 host selects among four SHA-addressed organs;
2. FERTIG verifies the result or handles a grounded fallback;
3. contradictions and unsupported structures return abstention.

No online learning step can mutate this path. This makes exact results usable
as certificates around neural inference.

## 9. Causal Prefix Sinkhorn Attention

CRSA is IMMER's Causal Prefix Sinkhorn Attention mechanism. It assigns heads
explicit causal roles:

- **Local:** bounded recent context;
- **Balanced:** prefix-mass-balanced Sinkhorn attention;
- **Free:** ordinary causal softmax.

The deployed role-complete program is two Local heads, one Balanced head, and
one Free head. The Free head preserves unrestricted causal reach and is
bit-exact with causal softmax for the same logits.

## 10. Persistent state

The O(1)-state runtime maintains a separate life stream with surprise-gated
updates, replay, and sleep consolidation. It can learn without rewriting the
frozen exact host or the immutable frontier checkpoint.

## Invariants

- Weight bytes are immutable.
- Graph appends are durable and independently verifiable.
- Atlas history proves exact sequence/event membership.
- Remote model sources require immutable revisions.
- Every range read is identity-bound and byte-accounted.
- Transport hints never change official router decisions.
- OoE executes only covered, calibrated, verifier-bound Crystal actions.
- Saved Qwen forwards commit only after final result verification.
- Exact capability execution either returns a verified answer or abstains.
- Private state and operational topology stay outside the public repository.
