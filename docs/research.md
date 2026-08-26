# Research Foundations

IMMER joins six research lines: sparse expert computation, contextual
sparsity, Causal Prefix Sinkhorn Attention, lifted Markov consensus,
reservoir-style state, and deterministic causal knowledge execution.

## David Tom Foss: causal knowledge and deterministic validation

- **The `.causal` Format: Embedded Deterministic Inference for Domain-Agnostic
  Knowledge Graph Amplification.** IEEE IRI 2026 full research paper and oral
  presentation. The official conference record is listed on
  [David Tom Foss's site](https://davidtomfoss.com/) and in the
  [IRI session record](https://davidtomfoss.com/service/iri2026-session-e2-nlp-sentiment-multimodal-reasoning/).
- **Deterministic Validation for Reliable LLM-Based Causal Knowledge
  Extraction.** Presented at ICECET 2026; the
  [official paper and talk record](https://davidtomfoss.com/talks/deterministic-validation-llm-causal-extraction/)
  describes deterministic validation of extracted causal relations.
- The public implementation lineage is
  [dotcausal](https://github.com/DT-Foss/dotcausal) →
  [o1-state](https://github.com/DT-Foss/o1-state) → LiveCausal in IMMER.

These works supply the central separation used here: immutable source data,
explicit causal wiring, deterministic inference, and provenance-bearing
results.

## Sparse mixture-of-experts

- Noam Shazeer et al.,
  [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538),
  2017. Foundational sparse MoE gating paper; preprint.
- William Fedus, Barret Zoph, and Noam Shazeer,
  [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://jmlr.org/papers/v23/21-0998.html),
  JMLR 23, 2022. Peer-reviewed.
- Nan Du et al.,
  [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://proceedings.mlr.press/v162/du22c.html),
  ICML 2022. Peer-reviewed.

IMMER keeps the model's official MoE decision intact and learns the transport
distribution around it.

## Contextual sparsity and routing

- Zichang Liu et al.,
  [Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time](https://arxiv.org/abs/2310.17157),
  2023. **Preprint.** It predicts contextual sparsity without retraining the
  base model.
- P. Batorski et al.,
  [MACRO: Markov Chain Routing of Transformer Layers](https://arxiv.org/abs/2608.05872),
  2026. Preprint on Markov policies for layer routing.

IMMER's Markov controller addresses a different layer of the stack: it predicts
next-layer expert payloads for transport while official DeepSeek routing and
all transformer layers remain unchanged.

## Lifted Markov consensus and Organism of Experts

The current OoE branch treats large-model execution as an expensive teacher
process whose verified local transitions can be consolidated into many small
site-bound Markov kernels. O1-State selects measurements, the
SemanticWeightAtlas binds them to exact `.causal` weight coordinates, and a
PS-Lifted push-sum fuses replicated transition evidence before quantized
Crystal publication. Mobile fading reservoirs and Möbius rapidity remain part
of the executable acceptance gate.

This is distinct from caching an answer and from replacing Qwen's internal
router. The learned alphabet selects runtime actions such as restoring a
charged continuation, invoking deterministic FERTIG, mounting an organ,
probing a coordinate, or returning to Qwen. Each Crystal remains pinned to its
model, coordinate, graph revision, schemas, coverage, calibration, and
verifier evidence.

## Reservoir computing and persistent state

- D. Verstraeten et al.,
  [An experimental unification of reservoir computing methods](https://www.sciencedirect.com/science/article/pii/S089360800700038X),
  Neural Networks 20, 2007. Peer-reviewed.
- Francis Wyffels, Benjamin Schrauwen, and Dirk Stroobandt,
  [Stable Output Feedback in Reservoir Computing Using Ridge Regression](https://link.springer.com/chapter/10.1007/978-3-540-87536-9_83),
  2008. Peer-reviewed conference chapter.
- Daniel J. Gauthier et al.,
  [Next generation reservoir computing](https://doi.org/10.1038/s41467-021-25801-2),
  Nature Communications 12, 2021. Peer-reviewed.

The O(1)-state branch uses a fixed recurrent substrate, controlled adaptation,
and compact learned readouts while keeping frozen exact execution separate.

## Attention and mathematical working drafts

The repository includes three working drafts:

- [`Causal_Prefix_Sinkhorn_Attention_v0.5_FULL.pdf`](../research/papers/Causal_Prefix_Sinkhorn_Attention_v0.5_FULL.pdf)
- [`causal_prefix_sinkhorn_v0.4_polished.pdf`](../research/papers/causal_prefix_sinkhorn_v0.4_polished.pdf)
Their release status is **research working draft**. The runtime evidence for
CRSA is reported separately in
[`BENCHMARKS.md`](BENCHMARKS.md).

## Storage format

- [safetensors](https://github.com/huggingface/safetensors) defines the safe,
  zero-copy tensor container used by the immutable weight plane.
- [dotcausal](https://github.com/DT-Foss/dotcausal) defines the public causal
  knowledge-format line. IMMER's causalized model bundle is a directory that
  combines unchanged Safetensors with a LiveCausal sidecar. The bundle is a
  directory-level system object rather than a renamed tensor file.

## Author record

[David Tom Foss](https://davidtomfoss.com/) ·
[ORCID 0009-0004-0289-7154](https://orcid.org/0009-0004-0289-7154)
