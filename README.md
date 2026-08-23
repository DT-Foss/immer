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
- **Frontier execution.** A complete DeepSeek-V4-Flash decoder runs from exact
  weight ranges with authenticated layer-level resume.
- **Grounded composition.** [FERTIG](https://github.com/DT-Foss/FERTIG)
  supplies deterministic parsing, verification, and explicit abstention.
- **Transport intelligence.** Label-free route observations support held-out
  Markov and placebo studies while the model's official router remains
  authoritative.
- **Independent mechanisms.** IMMER includes its own Causal Prefix Sinkhorn
  Attention line,
  persistent O(1) state, and digest-bound exact capability organs.

## Selected trial evidence

| Trial | Result | Scope |
|---|---:|---|
| DeepSeek-V4 exact decoder | 43/43 layers complete | fixed exact layer-major integration run |
| DeepSeek-V4 + FERTIG | 6/8 correct, 0 wrong | fixed eight-item GSM8K integration slice |
| SHIP-v6 exact core | 152/152 answers and routes | frozen host plus four exact organs |

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
