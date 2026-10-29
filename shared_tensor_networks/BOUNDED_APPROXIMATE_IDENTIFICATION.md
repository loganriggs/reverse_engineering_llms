# Qualitative stability of identifiable bilinear features

This is an elementary compactness consequence of the exact component theorem. It is not a claimed novel algebraic result, a numerical error bound, or an optimization guarantee.

## Setting

Fix a teacher with h quadratic features h_i(x)=(a_iᵀx)(b_iᵀx), d>=2h, and linearly independent2hlinear factors. Its T task functions span every unordered product h_i h_j, including squares. These are the assumptions in GENERIC_BILINEAR_COMPONENT_IDENTIFIABILITY.md.

Normalize teacher factors to Euclidean norm1, absorbing their former scales into readouts. Consider competitors with exactly h bilinear features g_i(x)=(u_iᵀx)(v_iᵀx), ||u_i||=||v_i||=1, and product-readout matrix V with Frobenius norm<=M, where M is fixed and finite. Assume M admits the normalized teacher (otherwise the convergence claim below can be vacuous). Inputs follow a fixed nonsingular full-support Gaussian; for notational simplicity use N(0,I).

Let L be the mean squared population error over all T task functions. Normalize each quadratic feature in Gaussian L2, and measure feature mismatch by

D(g,h) = min over permutations pi and signs s_i of
         max_i ||g_i/||g_i||_2 - s_i h_pi(i)/||h_pi(i)||_2||_2.

This removes sign, scale and permutation. It compares feature functions, not physical execution reuse.

## Claim

For every delta>0 there exists epsilon(delta,M,teacher)>0 such that every competitor in this class with L<epsilon has D(g,h)<delta.

Equivalently, along any sequence of bounded-readout competitors with population error tending to zero, the feature mismatch tends to zero.

## Proof

The parameter set is the product of2hunit spheres and the closed Frobenius ball ||V||<=M, hence compact. Polynomial coefficients depend continuously on parameters, so Gaussian population error L is continuous.

Feature normalization is also continuous throughout this set. For unit u,v and standard Gaussian X,

E[((uᵀX)(vᵀX))²] = 1+2(uᵀv)² >=1.

Thus no competitor quadratic has zero L2 norm. Every signed/permuted distance is continuous, and the finite minimum defining D is continuous.

For a fixed delta, restrict to the closed subset K_delta where D>=delta. It is compact. If it is empty, the claim is immediate. Otherwise L attains its minimum on K_delta. That minimum cannot be zero: the exact component theorem would force every zero-error competitor to match teacher features up to scale/permutation, implying D=0. Consequently the attained minimum epsilon is strictly positive. L<epsilon therefore excludes K_delta. This proves the claim.

The equivalent sequential statement follows either directly or by taking a convergent subsequence of any hypothetical counterexample with D bounded away from zero.

## Why this helps, and what it does not say

This formally bridges exact identification and sufficiently accurate approximation on a bounded, fixed-width family. It explains why controlling scale and preventing unbounded readouts is more than an implementation convenience in a robustness argument.

It supplies NO computable epsilon, rate, or teacher-independent threshold. Epsilon may be extremely small for ill-conditioned teachers, nearly dependent task heads or a large M. It does not justify the empirical cosine>.99threshold or imply that source MSE1e-3is adequate. It does not cover arbitrary overcomplete dictionaries, missing task directions, finite empirical interpolation, degenerate teacher factors, or input distributions supported on a lower-dimensional set.

The bounded-readout assumption is sufficient for this proof, not proved necessary. Unit factor normalization alone does not bound V: nearly dependent feature dictionaries can require large compensating coefficients. The observed raw-factor scale explosions can be absorbed into V when normalizing factors; the resulting readout budget must be reported rather than assumed benign.

For an unnormalized nonzero feature (aᵀx)(bᵀx), write alpha=||a||||b|| and g=(a/||a||)ᵀx times (b/||b||)ᵀx. Then h=alpha g, and product-readout coefficient W_ij becomes V_ij=alpha_i alpha_j W_ij. This preserves all task functions but can enlarge ||V||. Reciprocal rescaling a->c a,b->b/c leaves alpha and this normalized readout unchanged for positive c; arbitrary full feature scaling does not.

The remaining quantitative question is to bound epsilon using task-span separation, teacher-factor conditioning and a readout budget, and test whether those bounds predict observed recovery. The compactness theorem does not resolve it.
