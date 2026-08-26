# Formula Leverage

Release: 0.8.0 · 2026-08-26

IMMER's strongest mechanisms share one pattern:

\[
\text{state} \longrightarrow \text{structured projection} \longrightarrow
\text{small active set}.
\]

The runtime applies that pattern across attention, transport, world modeling,
memory, algebra, stored compute, novelty, causal addressing, and exact
capability execution.

## 1. Prefix-mass-balanced attention

For a causal non-negative attention matrix \(A\), the Balanced role uses a
causal prefix normalization:

\[
B_{ij}=\frac{A_{ij}}{\sum_{r\le i}A_{rj}}, \qquad j\le i,
\quad B_{ij}=0 \text{ for } j>i.
\]

Local, Balanced, and Free heads form a role-complete causal program. The Free
head preserves unrestricted reach; the specialized heads impose useful
structure without removing it.

## 2. Markov expert transport

The official router produces the active expert set. IMMER learns only the
transition law around that output:

\[
\hat P_\ell(j\mid i)=
\frac{N_\ell(i,j)}{\sum_k N_\ell(i,k)}.
\]

For a token row with current experts \(S\), the next-layer score is the
mixture

\[
s_{\ell+1}(j)=\sum_{i\in S}w_i\hat P_\ell(j\mid i).
\]

The full distribution supports adaptive \(k\), entropy-aware prefetch, and a
direct placebo comparison. It is a transport prior, never a replacement for
DeepSeek's gate.

## 3. Action-conditioned Markov world models

Authenticated one-step evidence accumulates a separate count tensor for every
action:

\[
C_{a,s,s'}=\sum_{e:\,(s,a)\to s'} w_e,
\qquad
P(s'\mid s,a)=\frac{C_{a,s,s'}}{\sum_j C_{a,s,j}}.
\]

Coverage, normalized entropy, and peak probability are independent execution
gates. For evidence threshold \(m_0\),

\[
\operatorname{conf}(s,a)=
\min\!\left(1,\frac{\sum_j C_{a,s,j}}{m_0}\right)
\cdot\max_j P(j\mid s,a)
\cdot\bigl(1-H_{\mathrm{norm}}(P)\bigr).
\]

A finite-horizon dynamic program composes the admitted rows into new plans; the
teacher supplies one-step transitions and zero multi-step labels.

PS-Lifted fusion exchanges sufficient statistics across the replica topology.
The self-calibrated continuation parameter starts from the Fiedler proposal

\[
p_c^{(0)}=
\operatorname{clip}\!\left(0.85-0.05\log\lambda_2,\,
p_{\min},p_{\max}\right)
\]

and selects the candidate with the largest measured lifted spectral gap on
bounded graphs. The fixed-\(p_c\) barbell needs 96 rounds in the current trial;
self-calibration needs 58, while the complete topology needs 28.

## 4. Continuous PS-Lifted reservoir

The fixed lifted recurrence carries temporal state:

\[
z_t=(1-\eta)z_{t-1}+
\eta\tanh\!\left(\rho T^\top z_{t-1}+W_{\mathrm{in}}x_t\right).
\]

Each replica learns ridge sufficient statistics \(X^\top X\) and \(X^\top Y\).
PS-Lifted consensus fuses those statistics, then solves one readout. The
delayed-state benchmark reaches 90.8359% fused accuracy against 50.4532%
with zero state and 54.2800% under circularly shuffled temporal labels.

## 5. Algebraic crystallization

Algebraic Crystals fit the coordinate system in which a family composes:

\[
\phi_{\mathrm{add}}(x)=\alpha x+\gamma,
\qquad
\phi_{\mathrm{mul}}(x)=\alpha\log x+\gamma,
\]

with a finite-cyclic branch using an invertible winding modulo \(n\). Admission
requires an exact family contract, score threshold, and best-versus-second
margin. After snapping new \(\phi\)-values back to group elements, the group
accumulator performs exact long composition. Additive, multiplicative, and
\(\mathbb Z_3\) families all execute length 12 exactly in the fixed trial; the
permuted placebo is rejected.

## 6. Authenticated stored compute

A ComputeCrystal is an affine, permutation, lookup, or row-stochastic Markov
operator. Homogeneous affine, permutation, and Markov chains admit exact
fusion:

\[
\mathcal K_{\mathrm{fused}}=
\mathcal K_r\circ\cdots\circ\mathcal K_2\circ\mathcal K_1.
\]

The ComputeChargeReceipt binds the source program, parent operators, fused
operator, exact fusion verification, and verifier receipt. The materialized
charged route binds that charge basis to the durable bank publication and graph
revision.

Discharge accounting is

\[
W_{\mathrm{released}}=
\max\!\left(W_{\mathrm{source}}-W_{\mathrm{live}},0\right).
\]

The current four-operator affine route executes as one operator on 128 values
created after charging, with 27,648 historical work units released and
maximum output delta 1.7764e-15.

### Charged prefix plus live residual

For a planned primitive sequence (K_1,ldots,K_n), let the deepest charged
terminal cover (K_1,ldots,K_m). Runtime execution is

\[
y=
K_n\circ\cdots\circ K_{m+1}
\left(K_{1:m}^{\mathrm{charged}}(x)\right).
\]

The joined receipt verifies the intermediate hash

\[
H\!\left(K_{1:m}^{\mathrm{charged}}(x)\right)
=H(x_{\mathrm{suffix\ input}})
\]

and sums source/live work across both VM receipts. The fixed route reuses two
charged steps, computes two suffix steps, releases 9,216 work units, and
matches primitive execution within 1.3323e-15.

### Constructive Birkhoff operator bases

Every admitted doubly-stochastic kernel is represented as

\[
P=\sum_{k=1}^{M}w_k\Pi_k,
\qquad
w_k>0,\quad \sum_k w_k=1,\quad
M\le (n-1)^2+1.
\]

Each iteration selects a perfect matching on the positive residual support,
subtracts its minimum matched mass, and records one permutation atom. Persistent
storage is \(O(Mn)\), not \(O(n!)\). Theta search stays inside the Birkhoff
polytope by construction:

\[
P(\theta)=\sum_k \operatorname{softmax}(\theta)_k\Pi_k.
\]

The fixed 8×8 trial uses 33 atoms under the bound 50 and reconstructs future
row-vector execution within 6.6613e-16.

### Verified demand intelligence

For materialized route (i), unseen-first UCB1 uses

\[
\operatorname{UCB}_i=
\frac{R_i}{N_i}
+c\sqrt{\frac{2\log\sum_jN_j}{N_i}}.
\]

Verified failure blocks the exact tuple `(route, graph revision, input ABI)`;
a later verified success on the same tuple restores eligibility. PPM predicts
from the longest observed route suffix inside an explicit episode. Outcome
arrival time never defines program order; the bound selection time does.
Co-occurrence prefetch scores

\[
\operatorname{coScore}(j\mid A)=\sum_{i\in A}C_{ij},
\]

and resident retention uses David Foss's structural decay

\[
\operatorname{keep}(i)=|R_i|e^{-\alpha\,\operatorname{age}_i}.
\]

Logical time replaces wall-clock time, pins are never evicted, and byte budgets
include route metadata, programs, Crystals, and charge receipts.

Verified execution closes the demand loop with

\[
r_{\rm success}=1+
\frac{W_{\rm released}}{W_{\rm source}},
\qquad
r_{\rm quality\ failure}=-1.
\]

An operational abort carries no quality reward. Its append-only abort event
removes the unfinished pull, so infrastructure failure cannot silently lower an
operator's empirical value.

### Exact guarded affine monoids

Stack, counter, fingerprint, Decimal-Horner, and group maps share one exact
transition law over declared integer or modular coordinate rings:

\[
T_{A,b}(s)=As+b,
\qquad
T_{A_2,b_2}\circ T_{A_1,b_1}
=T_{A_2A_1,\,A_2b_1+b_2}.
\]

Each action also carries phase and domain guards. Fusion combines the numerical
pair and retains every intermediate guard stage, preserving the partial-action
domain exactly.

For a stack with \(k\) slots,

\[
\operatorname{push}_x(s)=S_\downarrow s+Bx,
\qquad
\operatorname{pop}(s)=S_\uparrow s.
\]

Depth \(k\) is live and depth \(k+1\) enters the absorbing dead state. The
fixed trial recovers unique buried symbols at depth 53 inside a 64-slot stack.

For the context-sensitive language,

\[
c_1=\#a-\#b,\qquad c_2=\#b-\#c,
\]

with actions

\[
\Delta_a=(1,0),\quad
\Delta_b=(-1,1),\quad
\Delta_c=(0,-1).
\]

A phase register \(q\in\{A,B,C,\bot\}\) enforces monotone
\(A\rightarrow B\rightarrow C\); acceptance requires both counters zero and
\(q\neq\bot\). This rejects equal-count order confusers such as `acb` and
`abcabc`.

Rolling fingerprints use two modular registers per side,

\[
h_j\leftarrow rh_j+v(x)\pmod {p_j},
\]

plus left/right lengths, separator phase, and dead state. Modular equality is
only a candidate; exact byte equality produces the final verifier receipt.

Decimal parsing uses exact Horner recurrence

\[
h\leftarrow10h+d,
\]

under the strict grammar `[+-]?(0|[1-9][0-9]*)`. The committed capacity is 128
digits; digit 129 enters dead rather than overflowing a float.

### Contextual algebra selection

The meta-agent treats each verified algebra program as a contextual Thompson
arm. For context \(c\) and program \(a\),

\[
z_a\sim\operatorname{Beta}(1+s_{c,a},1+f_{c,a}),
\qquad a^*=\arg\max_a z_a.
\]

Every update reconstructs the exact parent router and deterministic Thompson
choice before changing a posterior. Behavioral MAP-Elites controls the active
Pareto catalog. Incompatible program schemas form a direct-product execution:

\[
F(x_1,\ldots,x_m)=
(F_1(x_1),\ldots,F_m(x_m)),
\]

with independent ABIs and one receipt joining the lane proofs. No sequential
ABI is asserted.

## 7. Novelty and structured kernels

For site patterns \(X\), query \(q\), and inverse temperature \(\beta\), the
novelty gate uses

\[
E_X(q)=-\operatorname{LSE}(\beta Xq)+\frac12 q^\top q.
\]

Execution requires both an admitted site-energy threshold and a sufficient
gap to the second-best site. The fixed trial accepts both in-distribution
queries and rejects all three out-of-distribution queries.

Structured action kernels use tensor-train/MPO factors only when the
recomputed source error fits the authenticated budget. The current structured
kernel stores 1,408 instead of 8,192 numeric bytes at 3.58e-16 relative
error. Rank-budget overflow selects the exact dense representation.

For graph-edge exploration, the Fiedler projector
\(\Pi_F=B_FB_F^\top\) removes sign and degenerate-basis ambiguity:

\[
\operatorname{nov}(i,j)=
\sqrt{\frac{(e_i-e_j)^\top\Pi_F(e_i-e_j)}{2}}.
\]

The proposal priority multiplies novelty by confidence and applies the direct
edge penalty only when the edge already exists. In the fixed barbell trial the
best missing bridge scores 0.409362; the best existing edge scores 0.120790.

## 8. Causal weight addressing

Let \(x=(m,\ell,e)\) identify logical model \(m\), layer \(\ell\), and expert
\(e\). Let \(L\) identify one exact Safetensors layout. The causal reader
implements

\[
R_L(x)=\{(f_r,o_r,n_r)\}_{r=1}^{q},
\]

where each tuple names a shard, absolute offset, and byte length. The layout
identity makes \(R_L\) fail closed under checkpoint drift. Live append extends
the address map while the tensor body stays immutable.

## 9. Exact capability projection

The frozen host routes an input state \(h\) into a small capability bank:

\[
o^*=\arg\max_o g_o(h),
\]

then executes the selected structural organ and asks FERTIG to verify the
result. Unsupported or contradictory structure maps to abstention.

A sealed ResultCell is the endpoint-specific limit of this path: it binds the
complete cold Qwen/FERTIG execution identity and can return that exact result
with zero Qwen forwards after final parity. It is not a substitute for the
value-general ComputeCrystal operator.

The real holdout discharges five authenticated cold Qwen forwards as zero live
forwards with exact raw-document and final-semantic parity. One frozen evaluator
call verifies the gold-correct output; FERTIG abstains cold and warm, fixing the
quality authority to evaluator verification plus parity.

## Experimental discipline

Every proposed shortcut must beat a matched control on held-out data and keep
the underlying model output invariant when it claims transport-only behavior.
The static embedding-to-value sketch failed this gate at 24% versus 32%
placebo and is absent from the runtime.

The release-safe attention derivation is recorded in
[`research/crsa/attention.md`](../research/crsa/attention.md).
