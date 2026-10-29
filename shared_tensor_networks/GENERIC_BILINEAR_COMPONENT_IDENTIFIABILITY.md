# Exact component identification from a complete task span

This is a direct derivation for the current model family. Its novelty relative to algebraic identifiability literature has not been assessed. It is an exact-function theorem, not a sample-complexity or optimization guarantee.

## Assumptions and claim

Let `h_i(x)=(a_iᵀx)(b_iᵀx)`, i=1,…,h, with real homogeneous quadratic features and d≥2h. Assume:

1. The 2h linear forms a₁,b₁,…,a_h,b_h are linearly independent.
2. The task functions span all r=h(h+1)/2 pairwise products h_i h_j, including squares.
3. A competitor has exactly h homogeneous bilinear features `g_i=(c_iᵀx)(d_iᵀx)` and linear task readouts over their pairwise products.
4. It reproduces every task as a polynomial function, equivalently with zero population L² error under a nonsingular full-support Gaussian.

Then `g_i=α_i h_{π(i)}` for a permutation π and nonzero scalings α. Individual linear factors are consequently identified up to swapping and rescaling, with feature scale absorbed by the task readouts. Physical caching versus recomputation is not identified.

## Proof: recover the quadratic subspace

Independence permits an invertible change of input coordinates such that `h_i=z_(2i−1)z_(2i)`. If d>2h, extend these forms to a full coordinate system. Let U span these h quadratics, and S span their r distinct product monomials; dim(S)=r.

**Square-root subspace lemma:** if a homogeneous quadratic g has g² in S, then g is in U.

Write `g=Σ_j A_j z_j² + Σ_{j<k} B_jk z_j z_k`. S contains no z_j⁴. Its coefficient in g² is A_j², so all A_j vanish. The coefficient of z_j²z_k² is now B_jk². S contains this monomial only for a designated teacher pair. All other B_jk vanish, so g is in U.

The competitor's product span has dimension at most r and contains the r-dimensional task span S, hence equals S. Thus every g_i² is in S, and every g_i is in U by the lemma. The g_i must span h dimensions: fewer independent quadratics could not generate r independent products. The hidden subspaces therefore coincide.

## Proof: recover individual bilinear axes

Within U, the symmetric matrix of `g=Σ_i α_i z_(2i−1)z_(2i)` is block diagonal, with blocks `[[0,α_i/2],[α_i/2,0]]`. Its rank is twice the number of nonzero α_i. Rank is preserved by the invertible input change.

A bilinear feature has quadratic-form rank at most two. Every nonzero bilinear feature inside U is therefore a multiple of one h_i. Since the h competing features span U, they select distinct teacher axes. This proves the claim.

The task rank also excludes fewer than h hidden quadratics in this product-readout family. More than h competing features would invalidate the dimension-equality argument; arbitrary overcomplete representations are not covered.

## Why the conditions matter

- Observing only h₁² leaves other features unconstrained; more examples of that one function do not supply missing task directions.
- Dependent factors permit mixing: h₁=x₀x₁ and h₂=x₀x₂ can be replaced by g₁=x₀(x₁+x₂), g₂=x₀(x₁−x₂). They are still bilinear and have the same complete product span, but are not feature-wise rescalings/permutations. The common factor violates independence.
- Overcomplete competitors can have additional unused or compensating features.
- Zero empirical loss is not zero polynomial error. Approximate population agreement also does not automatically imply feature proximity. A robust theorem would need factor/head conditioning bounds; the current sampled factor matrices have condition numbers from 9.16 to 43.10. Those numbers depend on the chosen factor scaling and are not claimed to be gauge-invariant robustness measures.
- Function identification does not determine physical execution reuse; see the graph-intervention study.

## Checks and learned models

All three shared teachers satisfy factor independence and task-head rank ten. The rank lemma is checked for all 15 nonempty feature subsets of each teacher, totaling 45 checks. Permutation/scaling positive controls pass. A separate audit computes Gaussian quadratic-function inner products directly from matrices using `E[(xᵀQx)(xᵀRx)]=2 tr(QR)+tr(Q)tr(R)`, rather than the producer's monomial representation, and checks the counterexamples through exact polynomial coefficients.

For trained shared students, absolute Gaussian L² cosine is maximized under a one-to-one feature assignment. Scaling and permutation are removed. The .95 threshold is a descriptive post-training diagnostic, not proof of exact recovery.

| Source examples/task | Initial mean matched cosine | Trained mean matched cosine | Trained components above .95 |
|---:|---:|---:|---:|
| 64 | 0.238 | 0.430 | 0/24 |
| 256 | 0.238 | 0.777 | 13/24 |

Each row comprises three teachers × two initializations × four components; the components are correlated, not 24 independent replications. This is partial recovery, not evidence the exact theorem has been reached by optimization.
