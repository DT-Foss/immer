# Formula Leverage

Release: 0.8.0 · 2026-08-23

IMMER's strongest mechanisms share one pattern:

\[
\text{state} \longrightarrow \text{structured projection} \longrightarrow
\text{small active set}.
\]

The runtime applies that pattern at four scales.

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

## 3. Causal weight addressing

Let \(x=(m,\ell,e)\) identify logical model \(m\), layer \(\ell\), and expert
\(e\). Let \(L\) identify one exact Safetensors layout. The causal reader
implements

\[
R_L(x)=\{(f_r,o_r,n_r)\}_{r=1}^{q},
\]

where each tuple names a shard, absolute offset, and byte length. The layout
identity makes \(R_L\) fail closed under checkpoint drift. Live append extends
the address map while the tensor body stays immutable.

## 4. Exact capability projection

The frozen host routes an input state \(h\) into a small capability bank:

\[
o^*=\arg\max_o g_o(h),
\]

then executes the selected structural organ and asks FERTIG to verify the
result. Unsupported or contradictory structure maps to abstention.

## Experimental discipline

Every proposed shortcut must beat a matched control on held-out data and keep
the underlying model output invariant when it claims transport-only behavior.
The static embedding-to-value sketch failed this gate at 24% versus 32%
placebo and is absent from the runtime.

The release-safe attention derivation is recorded in
[`research/crsa/attention.md`](../research/crsa/attention.md).
