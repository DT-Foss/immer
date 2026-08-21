# Corrected Prefix-Balance Composition Audit

## Verdict

The original experiment did not test the causal operator it described. Its
`clamp_min(finfo.tiny)` step converted masked future zeros into positive mass
and row normalization then allowed that mass to dominate for alpha >= 1.
After exact support preservation, the scalar family is **not a semigroup**:
no tested nontrivial pair closes to numerical precision, composition order
matters, and the best effective alpha is matrix-dependent.

- Exact closures: **0/49**
- Residual <= 5% of the composition effect: **14/49**
- Median relative closure residual: **0.0945**
- Maximum relative closure residual: **0.2583**
- Maximum order-commutator TV: **0.1272**

## Root-cause reproduction

With the legacy implementation at alpha=1, forbidden future mass after one pass is **114.514336** across the audit batch; at alpha=2 it is **379.999878**. The corrected implementation is exactly 0.

This also invalidates the old alpha=1 fixed-row and alternating/collapse claims:
they were properties of a dense, mask-breaking numerical operator.

## Best approximate closures

| a1 | a2 | a_eff | test row-TV | residual/effect | commutator TV |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.25 | 0.4483 | 0.0040 | 0.0187 | 0.0000 |
| 0.25 | 0.50 | 0.6473 | 0.0087 | 0.0275 | 0.0027 |
| 0.50 | 0.25 | 0.6511 | 0.0094 | 0.0294 | 0.0027 |
| 0.25 | 0.75 | 0.8471 | 0.0141 | 0.0332 | 0.0096 |
| 2.00 | 0.25 | 2.0174 | 0.0290 | 0.0349 | 0.0419 |
| 0.25 | 1.00 | 1.0473 | 0.0197 | 0.0371 | 0.0205 |
| 0.75 | 0.25 | 0.8595 | 0.0164 | 0.0380 | 0.0096 |
| 0.25 | 1.25 | 1.2485 | 0.0246 | 0.0393 | 0.0325 |

## Worst closures

| a1 | a2 | a_eff | test row-TV | residual/effect | per-matrix alpha std |
|---:|---:|---:|---:|---:|---:|
| 1.50 | 2.00 | 1.7755 | 0.2072 | 0.2583 | 0.0224 |
| 1.25 | 2.00 | 1.6784 | 0.1970 | 0.2525 | 0.0194 |
| 2.00 | 2.00 | 2.0943 | 0.2013 | 0.2369 | 0.0310 |
| 1.00 | 2.00 | 1.6433 | 0.1674 | 0.2178 | 0.0198 |
| 1.25 | 1.50 | 1.5603 | 0.1588 | 0.2124 | 0.0133 |
| 1.50 | 1.50 | 1.7042 | 0.1661 | 0.2121 | 0.0142 |
| 2.00 | 1.50 | 2.0845 | 0.1604 | 0.1899 | 0.0194 |
| 1.00 | 1.50 | 1.4687 | 0.1358 | 0.1892 | 0.0136 |

## Correct repeated-pass dynamics

Exact causal support is preserved and row entropy contracts monotonically in
the tested range. It does not alternate between future-mask artifacts.

| alpha | H(A) | H(P^10(A)) |
|---:|---:|---:|
| 0.50 | 3.1558 | 1.2465 |
| 1.00 | 3.1558 | 0.6512 |
| 1.25 | 3.1558 | 0.5159 |
| 1.50 | 3.1558 | 0.4195 |
| 2.00 | 3.1558 | 0.2869 |

## Descriptive parameter fits

These fits describe the best projection back into the one-parameter family;
they do **not** establish closure.

- Additive projection fit R²: 0.8165
- Multiplicative projection fit R²: 0.5245
- Asymmetric quadratic projection fit R²: 0.9923

## Reproducibility

- dtype: `float64`
- train/test matrices: 32/32
- sequence length: 96
- source SHA-256: `63a98d9a74c52db996424b710adc2fe43956f575e76f05ebd5fde7c8efb91836`
