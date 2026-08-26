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
  PS-Lifted Markov agents consolidate recurring runtime transitions into tiny,
  executable, quantized Crystals while Qwen remains teacher and novelty path.
- **Compute batteries.** Native Qwen continuation states are charged ahead of
  demand, restored by exact token prefix, and resumed without replaying the
  authenticated prefix.
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
| Qwen compute battery | 1.4288x peak speed | 65-token charged prefix + unknown 33-token suffix; authenticated restore removes 30.01% of demand latency |
| O1 → Atlas → OoE | crash-resumable live loop | real contextual Qwen receipts, append-only revision membership, partial Crystal promotion, zero-probe Atlas reuse |
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
