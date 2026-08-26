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

A finite-horizon dynamic program composes the admitted rows into new plans;
authenticated execution sources supply one-step transitions and zero
multi-step labels.

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

## 4. Consequence-grounded executable language

The sender and receiver retain separate policies:

\[
Q_S(a,w),
\qquad
Q_R^{G}(w,a),
\qquad
Q_R^{c}(c,w,a).
\]

The receiver sees only the opaque word \(w\) and authenticated context \(c\).
For local context visit count \(n_{c,w}\), its action score is

\[
\lambda_{c,w}=\min\!\left(1,\frac{n_{c,w}}{n_{\mathrm{full}}}\right),
\qquad
s(c,w,a)=(1-\lambda_{c,w})Q_R^G(w,a)
+\lambda_{c,w}Q_R^c(c,w,a).
\]

After the verifier returns scalar consequence \(r\), only the cells selected
by the sender and receiver update:

\[
Q\leftarrow Q+\alpha(r-Q).
\]

The receiver update contains no intent, target action, target state, or
semantic label. A word executes only when visit count, winning value, and
top-two margin all pass their gates. Shared evidence transfers stable meaning
into unseen contexts; the context residual learns a different action when the
same word has a different consequence in another state.

For a factorized message with slots \((w_1,\ldots,w_k)\), each slot owns its
sender and receiver tables. One whole-action consequence updates every selected
slot cell. Holding out Cartesian action tuples therefore tests recombination of
the learned factors rather than lookup of complete messages.

Executable definitions form a DAG:

\[
W_{d+1}=W_d\;b_d\;W_d,
\qquad
L_{d+1}=2L_d+1,
\qquad
L_d=2^{d+1}-1.
\]

Each level stores three direct references. Bottom-up Crystal fusion evaluates
every distinct DAG node once, preserves repeated parent hashes, and emits one
operator for the root. Transitive work provenance obeys

\[
E(K)=
\begin{cases}
W_{\mathrm{live}}(K), & K\text{ primitive},\\
\sum_i E(K_i), & K=K_m\circ\cdots\circ K_1
\text{ with provenance}.
\end{cases}
\]

The depth-12 trial binds 8,191 primitive actions to 36 references, releases
1,638,000 historical work units across 25 future states, performs 200 live work
units, and measures an 8.380x flat-versus-compiled speedup under the same
authenticated VM boundary.

## 5. Continuous PS-Lifted reservoir

The fixed lifted recurrence carries temporal state:

\[
z_t=(1-\eta)z_{t-1}+
\eta\tanh\!\left(\rho T^\top z_{t-1}+W_{\mathrm{in}}x_t\right).
\]

Each replica learns ridge sufficient statistics \(X^\top X\) and \(X^\top Y\).
PS-Lifted consensus fuses those statistics, then solves one readout. The
delayed-state benchmark reaches 90.8359% fused accuracy against 50.4532%
with zero state and 54.2800% under circularly shuffled temporal labels.

## 6. Algebraic crystallization

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

## 7. Authenticated stored compute

A compute battery moves already-performed work across time. Its stored unit can
be a final result, continuation state, operator, factorization, route, prefix,
search state, or residual state. `ComputeCrystal` is the numerical operator
form currently executed by the generic VM: affine, permutation, lookup, or
row-stochastic Markov. Homogeneous affine, permutation, and Markov chains admit
exact fusion:

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

### Prompt-preserving residual substrate

Whole-layer point fits discard the exact variable a causal model needs: token
order inside each prompt. The sequence instrument keeps every prompt separate
and predicts the residual

\[
\Delta_{p,t}=Y_{p,t}-X_{p,t}.
\]

After per-token RMS normalization, each fixed timescale advances

\[
r^{(\eta)}_{p,t}=(1-\eta)r^{(\eta)}_{p,t-1}
+\eta\tanh\!\left(W_r r^{(\eta)}_{p,t-1}+W_{in}\bar X_{p,t}\right).
\]

The readout features are

\[
\psi_{p,t}=\left[1,\bar X_{p,t},r^{(\eta_1)}_{p,t},\ldots,
r^{(\eta_m)}_{p,t},(r^{(\eta_1)}_{p,t})^{\odot2},\ldots\right],
\]

and ridge solves

\[
C=(\Psi^\top\Psi+\lambda I)^{-1}\Psi^\top\Delta,
\qquad \widehat Y=X+\Psi C.
\]

Recurrence resets at each prompt. Earlier prompts fit \(C\), one later prompt
selects feature semantics and \(\lambda\), and the final prompt is opened only
by the holdout evaluator. Raw X/Y content identity blocks the same sequence
from re-entering under renamed prompt/evidence hashes. The fixed synthetic
trial reaches MSE \(6.726089\times10^{-5}\) against tuned pointwise
\(5.803816\times10^{-2}\); token/output placebos reach \(3.1959/3.1541\).

### Left-acting causal sequence Crystals

For a realized causal kernel \(K\in\mathbb R^{T\times T}\),

\[
K_{qk}=0\quad(k>q),\qquad K\mathbf1=\mathbf1,
\qquad Y_{\ldots q}=\sum_k K_{qk}V_{\ldots k}.
\]

`CAUSAL_MIX_FLOAT64` fixes \(T\) as the final ABI axis and leaves batch, head,
and value-channel axes as independent applications. Two stored sequence
operators compose in execution order:

\[
K_{2\circ1}=K_2K_1.
\]

This is the native orientation of Attention value mixing. Its verifier and
work receipt are separate from Markov row-vector probability transport.

### Harvested-program routing

A contextual discovery becomes a routed program only after an exact bridge:

\[
H_{\rm bridge}=H(G_t,e,K,E_{\rm fit},E_{\rm holdout},V_{\rm discover},P),
\]

where \(G_t\) is the operator-graph head, \(e\) the selected edge, \(K\) the
published Crystal, and \(P\) its tagged `ComputeProgram`. Runtime verification
is a separate family-specific contract:

\[
V_{\rm execute}=H(g, s_{\rm src},s_{\rm dst},
\operatorname{ABI}_{\rm in},\operatorname{ABI}_{\rm out},
\operatorname{kind},\operatorname{checker}).
\]

This prevents a fit/holdout verifier from being reused as result-quality
authority and prevents unrelated families from sharing one verifier identity.
The selected program executes through the Crystal VM; its VM receipt and
consumer-verifier payload form one `VerifierBoundOutcome` that updates the
same contextual Thompson posterior.

The fixed three-family trial reaches \(174/180\) total, \(60/60\) late, and
\(30/30\) final correct choices. Circularly destroying the context signal
reaches \(49/180\), \(13/60\) late, and \(5/30\) final.

When parallel routes share endpoints, the selected path is the ordered edge
identity rather than only \((s_{\rm src},s_{\rm dst})\):

\[
r=H(e_1,\ldots,e_m,G_t),\qquad
y=\operatorname{dischargeExact}(r,x).
\]

Canonical cost planning remains available; exact discharge preserves an
algebra agent's explicit alternative.

## 8. Novelty and structured kernels

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

## 9. Causal weight addressing

Let \(x=(m,\ell,e)\) identify logical model \(m\), layer \(\ell\), and expert
\(e\). Let \(L\) identify one exact Safetensors layout. The causal reader
implements

\[
R_L(x)=\{(f_r,o_r,n_r)\}_{r=1}^{q},
\]

where each tuple names a shard, absolute offset, and byte length. The layout
identity makes \(R_L\) fail closed under checkpoint drift. Live append extends
the address map while the tensor body stays immutable.

## 10. Exact capability projection

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
