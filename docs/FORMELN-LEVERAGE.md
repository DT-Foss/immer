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
