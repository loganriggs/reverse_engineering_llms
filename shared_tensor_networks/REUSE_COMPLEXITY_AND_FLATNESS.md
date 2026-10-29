# An exactly solved reuse/optimization family

This elementary family connects directions A (minimal reusable computation) and B (gradient flow). It is a controlled example, not a claim that degree effects explain every deep bilinear network.

Fix binary addition/multiplication gates, scalar variable input t, arbitrary real constants, and fan-out allowed. For f(t)=t^m with m=2^D:

1. Its minimum number of multiplication gates is exactly D. Repeated squaring achieves D. After j multiplication gates the maximum degree of any computed polynomial is at most2^j: addition cannot raise the maximum degree, and multiplication adds the two input degrees. Thus degree2^D needs at leastD multiplication gates, regardless of extra additions.
2. Its minimum binary formula gate count is exactly m-1. A formula needs at leastm variable-leaf occurrences to obtain degree m, hence at leastm-1 internal gates; a product of m leaves achieves equality.

This is an exact resource certificate, unlike the earlier multivariate q^k circuit upper bound. Its univariate coefficient tensor has CP rank1; the multivariate q^k example supplies the separate large-flattening-rank certificate. Do not conflate them.

## Gradient flow is not controlled by circuit size alone

For zero-target squared loss L(t)=t^(2m)/2,

L'(t)=m t^(2m-1), L''(t)=m(2m-1)t^(2m-2).

For positive t0, ordinary Euclidean gradient flow has exact solution

t(s)=[t0^(-(2m-2))+m(2m-2)s]^(-1/(2m-2)).

Differentiate to verify dot(t)=-m t^(2m-1). The multiplicity-corrected flow removes the factor m in front of (2m-2)s. Expanding the repeated squaring into m occurrence parameters, f(z)=product_i z_i, gives identical occurrence gradients at z_i=t; their sum is the tied gradient. On this symmetric trajectory, untying offers no disagreement gradient. This differs from the image experiment, where occurrence gradients disagree substantially.

For fixed0<t0<1 and a fixed fractional reduction a in(0,1), the time to reach a*t0 is

[(a*t0)^(-(2m-2))-t0^(-(2m-2))]/[m(2m-2)].

It grows rapidly with m, although the minimum reusable circuit has only log2(m) gates. Here degree grows with circuit depth; this is not a fixed-degree causal demonstration that sharing harms learning. Absolute error is already tiny near zero, so slow relative parameter movement is not necessarily poor predictive performance. Untied product dynamics on the symmetric subspace share this flatness up to the multiplicity time factor.

## Explicit likelihood, not Hessian-based RLCT inference

Let Y|t ~ Normal(t^m,sigma^2) with known sigma>0 and truth mean0. KL is t^(2m)/(2sigma^2). With prior density smooth and positive near0 on a compact neighborhood, substitution u=n^(1/(2m))*t gives the local integral proportional to n^(-1/(2m)), hence local RLCT1/(2m). All m>=2 have zero Hessian at0, while these exponents vary with m. The specified likelihood and exact integral establish the exponent; Hessian rank does not.

check_reuse_flatness.py verifies derivatives, occurrence-gradient sum and the closed-form flow for D=1,...,6 in float64. The minimality and integral arguments are proofs; finite checks are implementation corroboration. There is no novelty or general-network RLCT claim.

Next useful empirical extension: multi-variable, fixed-degree targets with controlled repeated versus distinct factors, holding function scale and optimization metric fixed. Compare known sharing assumptions against mismatched assumptions. Keep this separate from varying depth/degree, which this example deliberately exposes as a confound.
