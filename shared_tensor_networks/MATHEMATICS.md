# Identification maps, geometry, and observational limits

This note gives direct derivations for the starting experiments, not a claim of a new theorem or literature survey.

## Pullbacks and the metric

Let an occurrence-space scalar loss be ℓ(z), with z=Sθ for a fixed linear full-column-rank map S. Chain rule gives

`∇θ ℓ(Sθ) = Sᵀ ∇z ℓ`, and `Hθ = Sᵀ Hz S`.

There is no extra second-derivative-of-S term because S is linear. For nonlinear identifications such a term generally exists. Input identification R is held fixed here.

The occurrence-space orthogonal projector onto im(S) is `P=S(SᵀS)⁻¹Sᵀ`. Projected Euclidean flow `ż=−P∇zℓ` corresponds to `θ̇=−(SᵀS)⁻¹∇θℓ`. Ordinary Euclidean tied-coordinate flow instead yields `ż=−SSᵀ∇zℓ`. If S simply repeats coordinates, SᵀS is diagonal with occurrence multiplicities. Unequal multiplicities cannot be corrected by one global learning rate. Discrete plain SGD followed by orthogonal projection gives the same corrected update; Adam's adaptive moments require separate treatment and do not inherit that identity automatically.

## Degree of block subproblems

A multilinear contraction is linear in each occurrence, not necessarily in a shared tensor. After identifying k occurrences of a parameter block, the output has degree at most k in that block (possibly lower through cancellations). Squared residual loss has degree at most 2k. A linear least-squares block update is justified only if the output is affine in that block.

For the depth-two bilinear network, the input has degree four. The early A matrix occurs in both h factors and has degree two; the squared residual can have degree four in A. C, D, and w each appear once and have linear output subproblems with other parameters fixed. In the untied occurrence network, each A_L, B_L, A_R, B_R appears once. These statements concern algebraic subproblems, not convergence guarantees for alternating minimization.

## Scalar singularities under an explicit likelihood

Take independent observations `Y=c+ε`, `ε~Normal(0,σ²)`, with known σ>0, and mean model μ(a,b)=ab or μ(t)=t². KL from the true data distribution is `K=(μ−c)²/(2σ²)`. The statements below assume a smooth positive prior in a bounded neighborhood of the point under consideration. Define the local exponent λ and pole multiplicity m by an integral scaling proportional to `n^(−λ) (log n)^(m−1)`.

For c>0, untied solutions ab=c are a one-dimensional smooth fiber. Locally the transverse KL is quadratic, giving λ=1/2, m=1. Tied solutions ±sqrt(c) are isolated but also regular in their single coordinate, yielding the same λ=1/2, m=1. Removing a flat fiber therefore need not change the exponent.

At c=0 and t=0, tied KL is proportional to t⁴. Substitution `u=n^(1/4)t` gives λ=1/4, m=1. At the untied origin, KL is proportional to a²b². Integrating over a bounded rectangle gives order `n^(−1/2) log n`, hence λ=1/2, m=2. At a nonzero point on either zero axis, it is regular transversely, with λ=1/2, m=1. The prior support and which singular points it includes matter. No network RLCT is estimated from a training Hessian here.

The scalar loss 0.5(t²−c)² has derivative 2t(t²−c) and second derivative 6t²−2c. At c>0, t=0 is a local maximum and ±sqrt(c) are minima. At c=0, zero is a quartically degenerate minimum. For untied loss 0.5(ab−c)², stationary points satisfy `(ab−c)b=(ab−c)a=0`: for c>0 the solution hyperbola plus the saddle origin; for c=0 the union of axes.

## Reuse is not identifiable from function observations alone

Suppose hθ(x) is deterministic and pure. A DAG that computes hθ(x) once and passes it to two consumers has exactly the same function as a formula that computes two copies using the same θ. Substitution proves equality, and differentiation of the same expression adds the same parameter-gradient contributions. Full-batch training using these derivatives agrees, up to numerical execution differences. Stochastic state, dropout, mutable buffers, precision, caching policies, or occurrence-specific regularizers can break this equivalence; these are additional experimental factors.

Consequently, outputs or input-output interventions alone cannot distinguish these two execution graphs. Even exact polynomial coefficient recovery is function recovery, not graph recovery. A graph complexity penalty selects a preferred explanation, not evidence that the selected graph was the physical teacher. E3 must specify a canonical/minimal equivalence class, an identifiable restricted graph family, or access to hidden states/interventions. Permutation/scaling symmetries are only part of the ambiguity: expansion, factoring, and common-subexpression elimination also preserve functions.

## Exact Gaussian function metric

Write the homogeneous quartic as `f(x)=Σα cα x^α`, where |α|=4. For standard independent Gaussian coordinates, `E[x^β]=Πj m(βj)`, with m(k)=0 for odd k, m(0)=1, and m(2r)=(2r−1)!!. Define `Gαβ=E[x^(α+β)]`. Then exact Gaussian MSE between coefficient vectors is `(c−c*)ᵀG(c−c*)`. Teacher RMS is `sqrt(c*ᵀG c*)`. The coefficient Euclidean metric is basis dependent; the Gaussian metric measures distributional function error.

The number of quartic monomials is binomial(d+3,4). Function recovery in this finite space does not identify a unique factorization, even if generic sample evaluations determine every coefficient.
