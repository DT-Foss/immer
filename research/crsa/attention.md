# Causal Prefix Routing: Release-Safe Research Note

This note records the public attention result behind IMMER's CRSA runtime. It
contains the operator, architecture rule, and measured research outcomes. The
integrated implementation lives in `src/immer/attention/`.

## 1. Causal Prefix-Sinkhorn

Let \(A\) be a non-negative causal attention matrix. Prefix-Sinkhorn balances
each visible key against the mass assigned to that key up to the current query:

\[
Q_{ij}=\frac{A_{ij}}{\sum_{r\le i}A_{rj}}, \qquad j\le i,
\quad Q_{ij}=0 \text{ for } j>i.
\]

The operator is token-to-token, strictly causal, and has zero dependency on a
future suffix. One normalization step is the canonical form measured in the
research sweep. Additional steps move too much mass toward local and diagonal
structure.

Measured column deviation was 0.105 for Prefix-Sinkhorn and 0.899 for causal
softmax on the fixed research setup, placing Prefix-Sinkhorn 8.6× closer to
the balanced target.

## 2. Width scaling

The fixed \(k=1\) sweep measured the following Softmax minus Prefix-Sinkhorn
gains:

| Width \(d\) | Gain, bpb |
|---:|---:|
| 128 | +0.984 |
| 256 | +1.106 |
| 384 | +1.173 |
| 512 | +1.182 |
| 1024 | +1.184 |

The curve rises through medium width and reaches a plateau near 1.18 bpb. The
measured mechanism therefore survives width scaling across this series.

## 3. The Free-head invariant

Fully specialized local or prefix-balanced attention can fail long-distance
copy tasks. Every tested route program that retained at least one ordinary
causal-softmax head recovered perfect Lag-48 copy accuracy in the fixed screen.

The architectural invariant is

\[
H=H_L\cup H_B\cup H_F, \qquad |H_F|\ge 1,
\]

where \(L\) is Local, \(B\) is Balanced, and \(F\) is Free causal softmax.

Three-seed byte-model results:

| Architecture | Test bpb | Gain vs softmax | Seed wins |
|---|---:|---:|---:|
| 3 Local + 1 Free | 4.720256 ± 0.015395 | −0.099746 ± 0.012807 | 3/3 |
| 2 Local + 1 Balanced + 1 Free | 4.726772 ± 0.010032 | −0.093230 ± 0.007312 | 3/3 |
| 1 Local + 1 Balanced + 2 Free | 4.735188 ± 0.007419 | −0.084814 ± 0.004023 | 3/3 |
| Softmax | 4.820002 ± 0.003515 | baseline | — |

A dedicated Self head was unnecessary in the winning programs. The residual
path already carries the current token.

## 4. Depth routing

The public architecture family separates lower-layer structure from
upper-layer retrieval. The main depth programs are

\[
(3L+1F)\rightarrow(2L+1B+1F)
\]

and its reverse. Lower Local heads extract short-range byte and morphology
features; upper Balanced heads regulate accumulated key use; every layer keeps
an unrestricted Free route.

The 46-run width/depth campaign found:

- width 32: CRSA delta −0.0333 bpb versus softmax;
- width 64: CRSA delta −0.1757 bpb;
- width 96: CRSA delta −0.2237 bpb;
- lower routed layers followed by upper softmax layers beat routing every
  layer in the tested deeper configurations.

This establishes Prefix-Sinkhorn as a foundation operator in the measured
depth series.

## 5. Long-horizon run

The 1,200-step, three-seed comparison produced:

| Architecture | Validation bpb | Delta vs softmax | Seed wins |
|---|---:|---:|---:|
| 3 Local + 1 Free → Softmax | 3.799582 | −0.250706 | 3/3 |
| 2 Local + 1 Balanced + 1 Free → Softmax | 3.811464 | −0.238824 | 3/3 |
| Adaptive LLL → Softmax | 3.813661 | −0.236626 | 3/3 |
| Adaptive LLB → Softmax | 3.814822 | −0.235466 | 3/3 |
| Softmax → Softmax | 4.050288 | baseline | — |

The variable-lag screen adds the robustness distinction:

| Architecture | Unseen accuracy | Worst lag | Mean bits |
|---|---:|---:|---:|
| Marginal Residual | 0.9363 | 0.8943 | 0.6262 |
| Fixed CRSA | 0.9293 | 0.9020 | 0.5872 |
| Softmax | 0.7745 | 0.6982 | 1.1481 |

Adaptive balance wins mean out-of-range accuracy; fixed CRSA wins worst-lag
accuracy and mean cross-entropy in this screen.

## 6. Anti-shortcut evaluation

Long-range evaluation must defeat recency and frequency shortcuts. The public
test design uses four task families:

- **buried-key recall:** older correct values compete with younger distractors;
- **random-lag multi-query copy:** every query receives a different lag;
- **pointer chase:** the answer requires multiple content-addressed jumps;
- **collision recall:** repeated values and similar keys eliminate frequency
  and position shortcuts.

A long-range result becomes headline evidence only when the softmax reference
learns the task and the specialized operator wins under the same solvable
setup.

## 7. Integrated runtime form

IMMER deploys the fixed role-complete program:

\[
2\ \text{Local}+1\ \text{Balanced}+1\ \text{Free}.
\]

The runtime tests enforce:

- exact causal support;
- zero future-suffix influence on prefix outputs;
- a complete unchanged Free head;
- deterministic role assignment;
- valid row sums and finite outputs;
- bit-exact equality between the Free head and ordinary causal softmax.

The public working drafts are:

- [`Causal_Prefix_Sinkhorn_Attention_v0.5_FULL.pdf`](../papers/Causal_Prefix_Sinkhorn_Attention_v0.5_FULL.pdf)
- [`causal_prefix_sinkhorn_v0.4_polished.pdf`](../papers/causal_prefix_sinkhorn_v0.4_polished.pdf)

Their publication status is **research working draft**. Runtime evidence and
its exact scope are recorded in [`../../docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md).
