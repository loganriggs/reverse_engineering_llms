# Generic fibers and a local Gaussian learning coefficient

This calculation concerns the shared multi-task architecture with d=8 inputs, h=4 quadratic bilinear features, and T=12 linear heads over their ten pairwise products. It is a local, realizable calculation at a generic parameter point. It does not estimate the learning coefficient of the trained approximate solutions or of a misspecified/overcomplete model.

For background on the distinction between ordinary parameter dimension and the RLCT in singular models, see Watanabe's [A Widely Applicable Bayesian Information Criterion](https://www.jmlr.org/papers/volume14/watanabe13a/watanabe13a.pdf). The model-specific calculation below is our direct derivation, not a result attributed to that paper.

## Eight continuous gauge directions

There are 184 parameters: A and B each have 4×8 entries, and W has 12×10 entries. For each hidden feature, independently rescale `a_i→u_i a_i` and `b_i→v_i b_i`, and compensate each product readout by dividing its coefficient by `(u_i v_i)(u_j v_j)`. There are eight independent local scalar freedoms.

Under the component-identifiability theorem's assumptions, the full exact-function fiber consists of these rescalings together with discrete feature permutations and factor swaps. Once hidden functions are fixed, product coordinates are independent and their task coefficients are unique. Thus the generic fiber has dimension eight. Discrete symmetries do not add continuous dimension.

## An exact Jacobian certificate

To avoid relying only on numerical singular-value thresholds, `exact_generic_rank.py` constructs the **integer polynomial-coefficient Jacobian** at a canonical point:

- `h_i=x_(2i) x_(2i+1)` for i=0,…,3.
- The first ten task heads select the ten products individually; two further heads are integer combinations.
- The map has 3,960 scalar output coefficients and 184 parameters.

Eight explicit independent integer gauge vectors lie exactly in its nullspace. Modular elimination gives rank176 using two moduli. The independent audit selects a 176×176 integer minor whose determinant is **714573 modulo1000003**, hence nonzero over the rationals and reals. The eight null vectors give the matching upper bound. Therefore the coefficient Jacobian has rank exactly176 at this point. Autograd derivatives of evaluations at fresh inputs agree with the coefficient-Jacobian evaluation to relative error1.59e-16.

The nonzero minor is a nonzero polynomial in the parameters, so rank176 holds on a nonempty generic open set. Gauge invariance bounds rank by176 on the open set of nonzero factors; polynomial minors then extend that upper bound globally. Rank can drop on an exceptional algebraic subset. This is an exact generic-rank statement for this architecture, not a claim that every sampled or trained parameter lies away from that subset.

## An explicit likelihood and local RLCT

Consider a separate realizable statistical model:

`X~N(0,I_8)`, `Y|X~N(f_θ*(X), σ² I_12)`, with known σ>0,

and the same mean-function family `Y|X,θ~N(f_θ(X), σ² I_12)`. Let θ* be a generic rank176 point and let the prior be smooth and strictly positive in a compact local chart around it. The true distribution here has Gaussian observation noise; this is not a reinterpretation of our noiseless optimization datasets as empirical RLCT measurements.

The population KL is exactly

`K(θ) = E_X ||f_θ(X)−f_θ*(X)||² / (2σ²)`.

The Gaussian moment inner product is positive definite on the finite polynomial coefficient space. Near the generic point, the constant-rank theorem supplies176 identifiable transverse coordinates u and eight fiber coordinates v. The mean function is locally constant along v. Its derivative in u has full rank, so the KL has a positive-definite quadratic leading term in u and is locally bounded above and below by positive multiples of ||u||².

Integrating `exp(−nK)` against the smooth positive local prior gives a finite positive contribution from the fiber coordinates and ordinary176-dimensional Gaussian scaling from u. Thus the local integral has leading order `n^(-88)`, with no logarithmic multiplicity factor:

**Local RLCT λ=88, multiplicity m=1.**

This is a locally regular quotient calculation. It is not obtained by blindly halving the rank of an arbitrary training-loss Hessian: it uses the specified likelihood, population KL, exact generic-rank certificate, constant-rank neighborhood, and positive local prior. The raw parameter count would suggest92; the eight gauge directions remove four from that regular penalty.

## What remains outside this calculation

Dependent factors, insufficient task span, vanishing components, overcomplete students, misspecified independent teachers, and globally noncompact rescaling fibers may change the analysis. Rank drops can create higher-order KL terms; there the Hessian rank is not sufficient to determine RLCT. The earlier scalar quartic example illustrates precisely that failure. No global marginal-likelihood exponent or causal explanation of the observed optimizer advantage is asserted here.
