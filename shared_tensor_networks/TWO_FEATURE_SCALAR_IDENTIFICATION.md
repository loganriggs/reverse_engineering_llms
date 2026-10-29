# A single output can identify two generic bilinear features

This is a restricted algebraic argument motivated by the readout-recovery experiment, not a claim of literature novelty or a theorem about arbitrary deep networks.

Let h1=(a·x)(b·x), h2=(c·x)(d·x), where a,b,c,d are linearly independent over the reals. Consider

f = alpha h1² + beta h1 h2 + gamma h2²,

with alpha*gamma != 0 and beta²−4 alpha gamma != 0. Among representations of f by two bilinear quadratic features and a homogeneous quadratic readout, the unordered feature lines are identifiable: any such representation has features proportional to h1,h2, up to permutation.

## Argument

Work first over the complex numbers. The binary quadratic readout has two distinct nonzero roots, so f factors as alpha(h1−r1 h2)(h1−r2 h2), with r1,r2 distinct and nonzero.

Use the four independent linear forms as coordinates. A pencil element u h1+v h2 has a symmetric matrix with two 2x2 off-diagonal blocks. Its determinant on these four coordinates is u²v²/16. If both coefficients are nonzero its rank is four; on either coordinate axis its rank is two. Extra input coordinates, if present, add zero directions. Therefore the only nonzero rank-at-most-two lines in this pencil are the h1 and h2 lines.

A homogeneous quadratic that factors into linear polynomials has symmetric matrix rank at most two. Hence both rank-four quadratic factors above are irreducible over the complex numbers. Unique factorization of complex polynomials fixes these two irreducible quadratic factors up to nonzero scalars and exchange.

Now suppose f has another two-bilinear-feature representation with features g1,g2. Its binary quadratic readout also factors over the complex numbers. Since f has exactly two distinct irreducible quadratic factors, those resulting quadratic factors must be the same two factors just identified. They are linearly independent and lie in span(g1,g2), so span(g1,g2)=span(h1,h2). Each gi is bilinear and therefore has rank at most two. The pencil rank calculation forces g1,g2 onto the two original feature lines. Since the features are real, their proportionality constants are real.

This identifies feature functions, not individual neuron parameter scales, the ordering of each feature's two linear factors, or the physical execution graph.

## Scope and empirical consequence

The proof fails if the four linear factors are dependent, either diagonal coefficient vanishes, or the discriminant vanishes. It provides no stability bound for approximate fits and no guarantee that training finds the exact solution. It also does not extend automatically to three or more hidden features: the scalar readout need not factor into two linear combinations of features.

Thus the d4/h2 experiment is an optimizer and supervision check in a family where one generic output already suffices for exact identification. Improved recovery rates with more labels would concern optimization robustness or stability, not the removal of a universal single-output ambiguity. This differs from the earlier d3/h3 non-recovery examples and the radial counterexample, which violate these assumptions.

`check_two_feature_identification.py` verifies the symbolic pencil determinant, ranks, an invertible nonorthogonal coordinate change, and a distinct-factor scalar example. The proof for all qualifying coefficients is the argument above; these fixtures alone would not establish it. The experiment audit separately checks teacher independence and readout nondegeneracy.
