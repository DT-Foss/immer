# Router Mathematics

Release: 0.8.0 · 2026-08-23

IMMER contains two distinct routers. They solve different problems and never
share authority.

## Causal Prefix Sinkhorn context router

The attention substrate is Causal Prefix Sinkhorn Attention. The context
classifier separates exact arithmetic requests from ordinary text. A frozen
host produces features \(h\); its trained classification head produces

\[
p(y\mid h)=\operatorname{softmax}(Wh+b).
\]

Its attention context is role-complete:

\[
H=H_{\mathrm{local}}\cup H_{\mathrm{balanced}}\cup H_{\mathrm{free}},
\qquad |H_{\mathrm{free}}|\ge 1.
\]

The deployed program uses two Local heads, one Balanced head, and one Free
head. The Free head equals ordinary causal softmax for the same logits.

Measured on the fixed 182-item arithmetic/text corpus:

- persisted context head: 182/182;
- raw frozen-host ablation: balanced accuracy 0.983333;
- 32 label placebos: mean 0.510328, maximum 0.748026;
- causal-softmax context ablation: 182/182.

The measurement proves the context feature and frozen router. It leaves CRSA
and causal softmax tied on this corpus.

## DeepSeek Markov transport router

For adjacent layers, each active token row supplies source expert set \(S_t\)
and target expert set \(T_t\). Counts accumulate without labels:

\[
N_\ell(i,j)=\sum_t \mathbf 1[i\in S_t]\mathbf 1[j\in T_t].
\]

Row-normalization gives the transition distribution. Current-row mixtures
produce a ranking over the entire target expert inventory. Evaluation sweeps
\(k\) and reports set recall, hit rate, precision, and predicted bytes.

Three controls are built into the evaluation:

1. **target marginal:** predicts frequent experts without using the current
   row;
2. **passthrough:** ranks the current expert IDs as the next-layer guess;
3. **placebo:** permutes aligned target rows while preserving layer marginals
   and row widths.

Prompt-level splits keep every row from one prompt in the same fold. The
controller can prefetch exact ranges for highly ranked candidates, but official
DeepSeek router output determines which experts execute.

## Adaptive transport width

A fixed top-k wastes bytes on confident transitions and misses broad
distributions. IMMER exposes the full score vector so the scheduler can choose
the smallest \(k\) that satisfies a probability-mass or resource target:

\[
k^*=\min\left\{k:\sum_{j\in\operatorname{TopK}(s,k)}s_j\ge \tau\right\},
\]

subject to current resident-memory and I/O budgets. The benchmark sweeps \(k\)
before any policy becomes a default.
