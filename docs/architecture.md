# Architecture

IMMER is a local runtime built from multiple execution planes. It composes
immutable neural weights, an append-only causal control plane, deterministic capability
organs, grounding, and persistent state behind explicit contracts.

## System map

```text
                              ┌─────────────────────┐
request ──> composition root ─┤ ExactCascade        ├─> answer / abstain
                              │ S3 organs + FERTIG  │
                              └─────────────────────┘
                                         │
                                         │ exact verification
                                         ▼
                              ┌─────────────────────┐
                              │ DeepSeek V4 runtime │
                              │ 43-layer decoder    │
                              └──────────┬──────────┘
                                         │ official router selections
                       ┌─────────────────┴─────────────────┐
                       ▼                                   ▼
             Markov transport hints              causal weight reader
             label-free, advisory                exact local ranges
                       │                                   │
                       └─────────────────┬─────────────────┘
                                         ▼
                              causalized local bundle
                              weights/ + causal/
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

## 3. DeepSeek-V4-Flash runtime

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

The layer stack is sequential because the transformer is sequential. The
causal graph accelerates address resolution and transport planning; it does
not replace the model's mathematics.

## 4. Markov transport controller

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

## 5. Exact execution plane

The `ExactCascade` owns exact arithmetic routing:

1. the frozen S3 host selects among four SHA-addressed organs;
2. FERTIG verifies the result or handles a grounded fallback;
3. contradictions and unsupported structures return abstention.

No online learning step can mutate this path. This makes exact results usable
as certificates around neural inference.

## 6. Causal Prefix Sinkhorn Attention

CRSA is IMMER's Causal Prefix Sinkhorn Attention mechanism. It assigns heads
explicit causal roles:

- **Local:** bounded recent context;
- **Balanced:** prefix-mass-balanced Sinkhorn attention;
- **Free:** ordinary causal softmax.

The deployed role-complete program is two Local heads, one Balanced head, and
one Free head. The Free head preserves unrestricted causal reach and is
bit-exact with causal softmax for the same logits.

## 7. Persistent state

The O(1)-state runtime maintains a separate life stream with surprise-gated
updates, replay, and sleep consolidation. It can learn without rewriting the
frozen exact host or the immutable frontier checkpoint.

## Invariants

- Weight bytes are immutable.
- Graph appends are durable and independently verifiable.
- Remote model sources require immutable revisions.
- Every range read is identity-bound and byte-accounted.
- Transport hints never change official router decisions.
- Exact capability execution either returns a verified answer or abstains.
- Private state and operational topology stay outside the public repository.
