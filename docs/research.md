# Research Map

IMMER connects research programs that solve different parts of the same system.

## FERTIG

Grounded neuro-symbolic cognition with a deterministic core, executable learned skills, constrained neural ranking, verification loops and process distillation.

Current structure-solver milestone: **1073/1319 GSM8K = 81.35% coverage,
246 safe abstentions, 0 incorrect and 0 error outcomes on the full test
split.** The report retains every item and pins dataset, runner and solver-tree
digests. An unverified operations-template fallback first produced 17/17
wrong attempts; it is now proposal-only and those cases are `must_abstain`
regressions. An earlier gold-label-free audit rejects the claim that its then
259 abstentions were ready assignment problems: the structural path found 144
unresolved pronoun/coreference cases and 115 unsupported grammar cases, with no
already extracted exclusive slot contract. Matching remains admissible only
after typed slot extraction and before the existing exact IR certificate.

## DeepSeek-V4 range runtime

The pinned 304B DeepSeek-V4-Flash checkpoint now executes through a local
43-layer, stateful decoder without loading the checkpoint as a resident model.
The path uses the original FP8/FP4 Safetensors, native compressed sparse
attention, hash/score MoE routing, HyperConnections, a bounded 12 GiB verified
range cache and the repository's fixed-role CRSA residual graft. Completed
integration runs prove one full autoregressive pass and a two-pass CRSA decode;
meaningful prompt quality and paired graft superiority remain open gates.

The exact LM-head transport now has four sealed alternating real-network
pairs: unchanged 127 compute blocks were carried by 16 rather than 127
physical range requests while every arm read 1,059,061,760 bytes. The full
blockwise FP32 logit hash and top-k values/IDs are bit-identical. The candidate
wins 4/4 pairs, with individual speedups from 1.375606× to 2.410811× and a
paired median of 1.445790×; mean time is 211.5756 s versus 125.6880 s. This is
repeated latency evidence, but transport width 1 remains the production
default until the end-to-end content gate. The evidence is
`results/deepseek-v4-head-range-network-smoke.json`.

## DeepSeek handoff falsification

The synthetic Wave3 demonstrations are not runtime evidence. T11 divides an
already averaged parallel gradient by `N` again and gives the sequential arm
privileged teacher intermediates; correcting the scaling leaves 5.19% rather
than 98% error reduction without repairing the target asymmetry. The 3-zone
router injects one synthetic `expert_quality` variable into gate norms and
expert embeddings, then recovers it through its score and oracle. Decoupling
leaves +4.67% over random and 36.66% of oracle; DeepSeek V4 has no matching
expert embeddings and uses a different official hash/score route.

Only two handoff ideas remain open as measurements: a passive residual trace
of the real HC-Sinkhorn recurrence, and a corrected per-head-temperature A/B
confined to the CRSA graft with exact causal masks and a permuted placebo.
Causal-adjusted Birkhoff quantities and offline ID/effective-rank estimates
may diagnose but not control execution. Zeno scheduling, Replica-MoE, a live
eta gate, Ginibre-Hurst control, mask recycling, SK1 and ID-based sizing remain
NO-GO until a new real-data mechanism and falsifier exist.

## Organ Grafting

`Organs, Not Weights` defines capability transplantation into a frozen living host.

Measured anchors from v0.1:

- host: **1,713,673 parameters**, 909,676,544 streamed tokens;
- donor: 27B, approximately **15,750×** larger;
- structured organs: **5,732–14,625 parameters**;
- host language NLL: **8.6656**, preserved through headline runs;
- recurrent chain organ trained at length 2: **1.000 through length 12** after crystallization;
- cold OrganBank Ship v6: **152/152 measured tasks**, **30/30 text routes**.

The central design rule is structure-first capability:

```text
STRUCTURAL FORM + MAP + ROUTE GATE + CRYSTAL
```

## Causal Prefix–Sinkhorn / CRSA

The v0.5 attention program establishes strict causal prefix-mass balancing and role-complete causal routing.

Measured anchors:

- exact zero future gradient and support for the causal operator;
- forced diagonal maximum at epsilon=0 and k=1;
- lower-triangular bistochastic obstruction: exact row+column stochasticity collapses to identity;
- k=1 remains the useful finite routing regime;
- scale gains: **0.984, 1.106, 1.173, 1.182, 1.184 bpb** at widths 128, 256, 384, 512, 1024;
- variable-lag unseen accuracy: **0.9293** for 2L+1B+1F→F versus **0.7745** softmax;
- every tested free-head-preserving lag-48 program: **1.000**;
- **69 automated tests** in the v0.5 program.

CRSA treats attention heads as routing roles rather than interchangeable copies of one operator.

## o1-state

The persistent host contributes the O(1)-state organism: constant-memory streaming, surprise-gated plasticity, external indexed memory, sleep/consolidation and portable state.

## FLCA

FLCA contributes typed operator evidence, compilation, schedule/runtime lowering and replay.

## QAD

QAD contributes the deployment axis. In the Organ Grafting deployment experiment, host fake-quant int4 self-distillation moved NLL from PTQ 8.6671 to 8.6654 against fp32 8.6656 while the tested organ capability remained intact.

## Primary References

### Peer-reviewed

- Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Maxim
  Krikun, Yuanzhong Xu, et al., 2022, [GLaM: Efficient Scaling of Language
  Models with Mixture-of-Experts](https://proceedings.mlr.press/v162/du22c.html)
  - large-scale sparse MoE language modeling with lower training cost than a
    dense baseline.
- William Fedus, Barret Zoph, Noam Shazeer, 2022, [Switch Transformers:
  Scaling to Trillion Parameter Models with Simple and Efficient
  Sparsity](https://jmlr.org/papers/v23/21-0998.html)
  - simplifies MoE routing to one expert per token and lowers communication
    cost.
- Daniel J. Gauthier, Erin Bollt, et al., 2021, [Next generation reservoir
  computing](https://doi.org/10.1038/s41467-021-25801-2)
  - modern reservoir-computing framing with linear training on the readout.
- D. Verstraeten, B. Schrauwen, M. D'Haene, D. Stroobandt, 2007, [An
  experimental unification of reservoir computing methods](https://www.sciencedirect.com/science/article/pii/S089360800700038X)
  - early unifying ESN/RC experimental reference.
- Francis Wyffels, Benjamin Schrauwen, Dirk Stroobandt, 2008, [Stable Output
  Feedback in Reservoir Computing Using Ridge Regression](https://link.springer.com/chapter/10.1007/978-3-540-87536-9_83)
  - ridge-regression readout as a stable reservoir-training primitive.
- Georg Holzmann, et al., 2010, [Echo state networks with filter neurons and a
  delay&sum readout](https://www.sciencedirect.com/science/article/pii/S0893608009001580)
  - ESN readout variants that keep the reservoir fixed and train the output
    layer.

### Preprints and specs

- Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le,
  Geoffrey Hinton, Jeff Dean, 2017, [Outrageously Large Neural Networks: The
  Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
  - baseline sparse MoE routing with a learned gating network over many
    experts.
- Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song,
  Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Ré, et al.,
  2023, [Deja Vu: Contextual Sparsity for Efficient LLMs at Inference
  Time](https://arxiv.org/abs/2310.17157)
  - contextual sparsity predictor for faster inference without retraining the
    base model.
- P. Batorski, et al., 2026, [MACRO: Markov Chain Routing of Transformer
  Layers](https://arxiv.org/abs/2608.05872)
  - Markov-policy baseline for layer routing with skip/repeat choices.
- [safetensors/safetensors](https://github.com/safetensors/safetensors)
  - safe, zero-copy tensor storage format; the format anchor for local range
    loading.
- [DT-Foss/dotcausal](https://github.com/DT-Foss/dotcausal)
  - the `.causal` format repository and public reference for the causal graph
    substrate.
