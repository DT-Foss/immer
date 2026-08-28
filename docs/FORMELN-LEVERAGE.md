# Formula Leverage

Release: 0.8.0 · 2026-08-27

Integration base:
`45778a4aeac55a8f1a7bcb47df73fee4972e1228`.

IMMER's strongest mechanisms share one pattern:

\[
\text{state} \longrightarrow \text{structured projection} \longrightarrow
\text{small active set}.
\]

The runtime applies that pattern across attention, transport, world modeling,
memory, algebra, stored compute, novelty, causal addressing, and exact
capability execution.

## 0. Learned predictive-state quotient

Histories belong to the same executable state exactly when their future laws
agree:

\[
P(Y_{future}\mid h)=P(Y_{future}\mid q(h)).
\]

The quotient is learned from exact empirical outcome counts and serialized in
canonical order. The first mechanism controls are decisive:

- a four-history process with two equal future laws contracts `4→2`;
- the non-bisimilar control remains `4→4`;
- a rational `2/3`–`1/3` law survives the merge exactly;
- held-out total variation is `0`;
- a disturbed future law has total variation `1`;
- reversed observation order produces byte-identical quotient state.

The live controller uses an agent-local transition law:

\[
(s,a)\longrightarrow(s,a_{executed}),
\]

where the site `s` remains the same. On all 16 live FERTIG learning receipts,
nine observed `(site, source-action)` states have one identical future law and
therefore contract `9→1`: `9x` compression after one refinement round, with
zero terminal and zero censored observations. The quotient SHA is
`064dd640c673a3e86dca24513663a1c3a7d2793ed279d2b0b9d2667081e15ae1`;
the persisted file SHA is
`44e3853cc9bb1953feb01146f4b634a75247cffdc5e129b80e6d7df7c5eaa3fa`.
Replacing that successor with global trace topology leaves all nine states
separate (`9→9`). This matched control proves that the compression is the
site-local Markov quotient, not a generic merge.

This quotient is the compression gate for language contexts, O1 histories,
route-demand episodes, and battery compatibility domains. It preserves the
future distribution that matters and deletes historical distinctions that do
not.

The next contraction is its conditional-independence blanket:

\[
Y \perp X_{\setminus B}\mid B,
\qquad
P(Y\mid X)=P(Y\mid X_B).
\]

IMMER searches the complete bounded feature power set with exact rational
probabilities and group-macro Brier scores. Truncated searches explicitly
abstain; singleton and pair checks remain diagnostics, never a proof against
higher-order XOR-like synergy. In the Demand route world this recovers the
non-contiguous causal lags `{1,3}` and safely precedes PPM/UCB. The complete
receipt replay scores Selected/Full `9/10`, contiguous PPM `4/5`, Random
`1/2`, and Marginal `2/5`, while four verified discharges release `144/336`
historical work units.

For Qwen boundaries, the learned blanket crystallizes the residual equation

\[
h_{layer\_out}=h_{attention\_residual}+h_{mlp\_out}.
\]

The fixed formula is selected from six authenticated boundary candidates,
holds on an unseen layer with NRMSE `0.0016575120`, and is selected in every
leave-one-question-out fold. Separately, exact-feature router blankets prove
that only the global top two centroids affect radius/margin routing, reducing
live distance work by `75%` without changing one decision.

Cross-layer/head transport now has an executable diagnostic in the causal
attention basis:

\[
C A = B C,
\qquad
\operatorname{vec}(CA-BC)
= (A^\top\otimes I-I\otimes B)\operatorname{vec}(C).
\]

IMMER fits the identifiable causal mass-preserving class, then rejects rank
deficiency, a missing identification gap, excessive system condition, or a
singular transport. It reads Prefix-Sinkhorn operators before native head
blending; archive Softmax matrices are not compatible evidence.

The MLP battery key is likewise joint rather than gate-only:

\[
K_{k,b}(h)=Q_b\!\left(
[\,W^{gate}_{I_k}h\;\Vert\;W^{up}_{I_k}h\,];s_{train}
\right).
\]

`I_k` and the symmetric channel scales are learned from training groups only.
A dimension/quantizer pair can promote only when train, calibration, and
holdout contain zero wrong-output collisions and at least one exact verified
hit.

O1 now captures the missing evidence without changing either formula:

\[
R_{h,q,:}=\operatorname{PrefixSinkhorn}(L)_{h,q,:}
\quad\text{before native blend},
\]

and

\[
(h,\;W^{gate}h,\;W^{up}h,\;W^{down}
(\operatorname{SiLU}(W^{gate}h)\odot W^{up}h)).
\]

Both are stored as measurement-bound binary sidecars. Atlas and O1 status carry
only their receipt hashes; exact tensor bytes never become semantic labels or
status payloads.

The MLP formula now has its full local-Qwen measurement path. The runner uses
IMMER's causal attention and captures only layers
`(45,18,36,63,27)/(0,9)/(54)` for train/calibration/holdout. Five prompt
forwards per split execute `645` layers in total. One weight-local
`linear_many` replay per gate, up, and down projection at each selected layer
gives `24` reads and proves the captured values against the immutable weights.
The bank invokes that verifier, stores the proof under CAS, and binds all 40
groups to one manifest, ModelPin, verifier, and exact O1/Atlas/Harvester
authority inventory. The `25/10` fit inventory is sealed before the five
holdout groups execute.

The official local Qwen3.8-27B measurement is complete. The smallest promoted
joint key selects intermediate coordinates `[6844,3028]` and concatenates
their gate/up values, producing four quantized scalars per token. At 16 bits it
records `232/1024` all-row calibration hits and `116/512` all-row layer-54
holdout hits with zero wrong collisions. Its `126,760 B` cache is `1,743.80x`
smaller than the full 17,408-coordinate 16-bit control (`221,043,712 B`) with
the same held hit/miss counts. Exact prompt-row roles then remove the 29-token
shared prefix and 9-token suffix: content-only calibration and L54 contain no
repeated output payload at all (`0/644` and `0/322` adaptive hits). Candidate
`k=1/16` remains correctly blocked by prior wrong collisions, but the surviving
`k=2` hits are template reuse rather than content-compute savings. The next
mechanism predicts layer-local Gate×Up ranges instead of exact output payloads.

That range mechanism is now measured. For layer \(\ell\), token \(t\), and
64-neuron block \(b\), the teacher energy is

\[
e_{\ell tb}=\sum_{j\in b}
\left(\operatorname{SiLU}(g_{\ell tj})u_{\ell tj}\right)^2.
\]

Residual OMP builds a block-local Markov blanket of four pilot neurons. At
step \(r\), it selects the neuron maximally correlated with the current
train-only residual, refits the ridge kernel, and repeats. Runtime computes all
pilots once, chooses the top 32 predicted blocks, and executes the union

\[
S_{\ell t}=P_\ell\cup\bigcup_{b\in\operatorname{Top32}(\hat e_{\ell t})}b.
\]

The union has `3,008/17,408` neurons (`17.279%`). On the completely later
five-prompt generation it captures `39.466%` activation energy versus
`27.754%` for an equal-compute static layer table and `25.890%` for random
pilots. The native BF16 kernel executes the pilot contribution and selected
block residual as two exact passes. Across `14,499,840` external output values,
cosine is `0.8790` and relative L2 error `0.4995`; the static arm gives
`0.7590` and `0.6947`. The next formula fits the omitted-output residual using
generation-1 evidence only.

That correction is the diagonal affine Organ

\[
\tilde y_{\ell t}=s_\ell\odot \hat y_{\ell t}+b_\ell,
\]

where each coordinate of \((s_\ell,b_\ell)\) is the closed-form OLS solution
from generation 1. The fit contains only `2×5,120` scalars per layer (`860 KB`
for all eight measured layers). Unchanged on generation 2, it raises cosine to
`0.94524`, lowers relative L2 to `0.32638`, and restores `89.49%` output
energy. Runtime cost is 5,120 multiplies plus 5,120 additions per layer.

The physical layout closes the compute loop. Gate/Up block rows already lie
contiguously in the checkpoint. `down_proj` is stored row-major in the opposite
orientation, so the local causal rail materializes the immutable view
\(W_{down}^{\mathsf T}\) once. Fixed pilot rows are packed once; dynamic actions
then read full 64-row blocks and zero the four duplicate pilot positions. The
executed byte fraction is

\[
\frac{272\cdot4+32\cdot64}{17,408}=0.180147.
\]

Across layers `0,9,18,27,36,45,54,63`, measured MLP time drops from
`3.419 s` to `2.289 s` (`1.494×`) and weight reads from `4.278 GB` to
`770.7 MB`. The first scattered layout required 2,366 original causal reads;
packing reduces this to 48–58 per layer without changing the selected action.

Mounted continuation confirms that local layer gains survive the surrounding
model. Starting from the same exact native prefix state on five unseen prompts,
the sparse and full decoders choose the same first token `5/5`; mean Top-10
overlap is `8.4/10` and final-hidden cosine is `0.9466`. With only eight of 64
layers currently measured, whole-decode time is `320.06 s` full versus
`318.58 s` sparse, while physical weight transport falls from `243.53 GB` to
`225.99 GB`. O1 coverage expansion, not another router formula, is now the
speed bottleneck.

The next selector uses a finite Markov walk rather than one static ranking:

\[
B_{t+1}=B_t\cup\{j_t\},\qquad
j_t=\arg\min_j\bigl(C_{wrong},-C_{exact},C_{miss},\text{bytes}\bigr),
\]

where every action is replayed by the exact collision cache on train-only
groups. Energy, Fisher discrimination, variance, and deterministic exploration
only nominate actions; they never authorize a basis. Calibration chooses among
the frozen beam, and the collision-capacity guard
`2 * |B| * quant_bits >= 64` prevents a superficially perfect 32-bit `k=1`
state from promotion. Holdout now reports both a frozen cache and the intended
online-adaptive cache explicitly.

The first real operator wave isolates one admissible directed pair, `20→8`.
Its held-out intertwining residual is `0.0248507190`, compared with
`0.2373913654` for `C=I` and `0.2729029466` for the deterministic random
orthogonal placebo. All other directed pairs fail invertibility/conditioning
before the holdout is opened.

The next formula-derived stack is:

\[
P(y\mid q,G)\approx P(y\mid q,C_q),
\]

where `C_q` is a learned conditional-independence Markov blanket, followed by
exact cross-basis operator transport

\[
O_B=C\,O_A\,C^{-1}
\]

under explicit ModelPins and basis receipts. The Qwen-native MLP battery then
learns a joint quantized subspace for `gate_proj` and `up_proj`; it never
reduces the gated MLP to a single projection. Topology-aware Warmth is a
downward charging brake: high local reuse suppresses redundant charging and
cannot manufacture demand.

The external formula source was audited at SHA
`5b986886988fe1a1c42256bdebaa6aff9b91588041f2af010ef80408a0b4e0b7`:
`214` normal tests pass in `148.20 s` with `560` warnings, while warning-fatal
execution passes `24/42` files and fails `18/42`. IMMER implements the formulas
in its own runtime. The source archive's fixed-`q,K` Softmax Attention code is
excluded because IMMER uses causal Prefix-Sinkhorn Attention.
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

### Continual frontier growth

One learned action row transfers from frontier \(F_t\) to \(F_{t+1}\) exactly
when all semantic authorities survive:

\[
\operatorname{retain}(a)=
\mathbf 1[B_t(a)=B_{t+1}(a)]
\mathbf 1[R_t=R_{t+1}]
\mathbf 1[\Pi(F_t\rightarrow F_{t+1})=1],
\]

where \(B(a)\) is the complete action/artifact binding, \(R\) is the hashed
reward policy, and \(\Pi\) is a typed transition proof. For route frontiers,
\(\Pi\) embeds every graph-state payload and verifies each append transition,
endpoint authority, and retained/added route inventory. Without that proof the
old row resets.

Context evidence has its own independent gate:

\[
Q_{R,t+1}^{c}=Q_{R,t}^{c}
\quad\Longleftrightarrow\quad
H(C_t)=H(C_{t+1}),
\]

so a context-schema shift drops local tables while globally stable word meaning
survives. Vocabulary capacity grows deterministically:

\[
|V_{t+1}|=max(|V_t|,|A_{t+1}|),
\]

with old rows copied on the exact \(|V_t|\) slice and new words initialized
unvisited.

Verified macro discovery operates on selection-ordered positive episodes. A
candidate word program \(u\) is admitted only when

\[
\operatorname{support}(u)\ge m,
\qquad
\operatorname{semantics}(u)=\{(a_1,\ldots,a_k)\},
\]

the declared episode length is complete, and a frontier-authorized terminal
verifier seals the episode. Compilation, action promotion, frontier migration,
and state publication form one revision receipt.

In the five-seed growth trial, old-action accuracy remains 1.0, the promoted
action begins behind abstention, becomes executable after 328 episodes on
average, and final four-action accuracy returns to 1.0. A no-memory model scores
0.0. A second-generation word uses the promoted macro twice and raises released
historical work from 400 to 1,200 units.

### Live controller-Crystal language

For promoted controller site (s), the stored uint16 kernel is restored as

\[
K_s=\frac{Q_s}{L},
\qquad
\sum_j K_s[i,j]=1,
\]

then published as a row-Markov ComputeCrystal only when identity-basis
execution reproduces the source kernel exactly:

\[
\operatorname{Crystal}_s(I)=K_s.
\]

The export authority is the joint digest of controller snapshot, model pin,
weight and Atlas revisions, calibration, source manifest, verifier/evidence
sets, quantized kernel, and append-only ComputeBank anchor. A language outcome
for intended site (i), selected site (s), and state vector (x) receives

\[
r(i,s,x)=
\begin{cases}
+1,&i=s\land H(C_i)=H(C_s)\land H(xK_i)=H(xK_s),\\
-1,&\text{otherwise}.
\end{cases}
\]

The receiver sees only its opaque word and context; the verifier replays both
artifacts. On the first live eight-action frontier, frozen vocabulary accuracy
is 1.0 and post-persistence greedy execution is 1,000/1,000. Three verified
repetitions of a four-action word program compile to one Crystal. Across 25
future vectors it releases 3,375 historical work units with maximum flat-chain
error (1.110223\times10^{-16}).

### Dialect quotient and portable programs

Two opaque words from independently learned dialects are equivalent only
through the executable consequence they acquired:

\[
w_i \sim w_j
\quad\Longleftrightarrow\quad
a_i=a_j
\;\land\;
B_i(a_i)=B_j(a_j)
\;\land\;
H_i^{\mathrm{authority}}=H_j^{\mathrm{authority}},
\]

where \(B(a)\) is the complete action/artifact binding. Translation therefore
operates on the quotient \(V/{\sim}\), not on token spelling. Before execution,
the translation receipt is reconstructed from both frozen snapshots and both
frontiers; a handcrafted surface permutation cannot become authority.

A portable word program stores

\[
P=(a_1,\ldots,a_k;
B(a_1),\ldots,B(a_k);
H_{\mathrm{discovery}},H_{\mathrm{definition}},E),
\]

with supporting trajectory set \(E\). Target dialect \(j\) localizes it as

\[
D_j(P)=
(\operatorname{encode}_j(a_1),\ldots,
 \operatorname{encode}_j(a_k)),
\]

only after every target binding and verifier authority equals its source
contract. The localized definition and lexicon state both bind the target
context before compilation in the target bank.

Across five seeds, five dialects, and three independently learned contexts per
dialect, quotient translation scores 60,000/60,000 unseen programs and 100% in
the worst target context. All four target dialects have three distinct context
mappings and zero globally stable action words. The shared-token direct baseline
scores 6.84% program accuracy while a non-identity permutation placebo scores
1.42%. The same portable macro localizes and executes in all 75
dialect-context targets; changed bindings and authorities are rejected.

The first live instance uses two independently seeded languages over eight
promoted Qwen/O1 site policies. Their action-word maps disagree on all eight
actions, while quotient translation is exact on all eight. Three isolated
support chains authorize a four-action portable program; its sibling
localization recompiles to one charged Crystal, releases 3,375 historical work
units across 25 future vectors, and stays within (2^{-52}) of flat execution.

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
