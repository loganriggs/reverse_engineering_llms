# Generic local identification with one scalar readout, d6/h3

The three-feature control has h_i=(a_i·x)(b_i·x), i=1,2,3, and each scalar output is a linear combination of the six unordered products h_i h_j. Six input variables and independent linear factors are used. This note gives a local result, not a global uniqueness or optimization theorem.

## Certificate

At a shared small-integer hidden-factor fixture, exact coefficient-Jacobian ranks are:

| Observations | Parameters | Rank | Scaling freedoms |
|---|---:|---:|---:|
| One head | 42 | 36 | 6 |
| Three heads | 54 | 48 | 6 |
| Six heads | 72 | 66 | 6 |
| Base head + three labeled ablations | 42 | 36 | 6 |

`certify_readout_local_rank.py` stores all matrices, parameters, pivot rows/columns, and nonzero minors modulo the prime2147483647. Determinants are1388055056,1485723961,1686740030,1201750359 respectively. Primality is checked by trial division. Small integer polynomial operations, including central differences at offsets1and2, are exactly representable in float64: entries use factor coordinates bounded by5, quadratics bounded by50, and sums over at most6products and21²monomial pairs, well below2^53. Every analytic derivative matches both exact central differences. Each individual parameter has degree at most two, making those differences exact rather than an approximation. Modular nonzero minors certify real rank lower bounds without a floating-point threshold.

## Why six dimensions must remain

For each hidden row, reciprocal rescaling a_i→c_i a_i, b_i→b_i/c_i preserves h_i. Independently rescale h_i by t_i and divide every readout coefficient for h_i h_j by t_i t_j (by t_i²on its diagonal). These give six independent continuous scaling directions at nonzero generic factors. The same transformations preserve labeled ablation outputs because multiplication by fixed masks commutes with the readout rescaling. The script independently verifies six linearly independent integer tangent generators annihilated by each coefficient Jacobian.

Consequently generic rank cannot exceed parameter count minus six. Each certified minor attains that upper bound. Since the coefficient map is polynomial and the minor is not identically zero, the same maximal rank holds outside a proper algebraic subset. In that regular region the local fixed-output fiber has dimension six. The six-dimensional scaling orbit occupies that fiber locally, so there are no additional local continuous degrees of freedom beyond these gauges.

This is generic **local** identification modulo scaling. It does not exclude discrete permutations, exchanges of the two factors within a feature, or distant inequivalent decompositions. It does not establish global identification from one head, nor identify physical caching/reuse. The integer fixture is not a certificate that every random trained point or teacher is regular. No claim of literature novelty is made.

## Implication for the experiment

Three-head and intervention supervision can help recovery even when a scalar head already supplies enough local constraints generically. The possible mechanisms include conditioning, finite-precision/noisy stability, and optimization basins. The six-head full-product-span theorem remains stronger about global feature identification under its assumptions; local rank does not replace it. The next informative test is stability of weak parameter directions under controlled perturbations, rather than treating a failed fit as proof that the observations are insufficient.
