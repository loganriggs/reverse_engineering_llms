# Similar states need not implement similar local computations

M100 studies native MLP1's context-dependent Jacobian, rather than fitting another output dictionary. M77 showed that a recipient-local operator predicts changes better than the training-mean operator. M79 tested Euclidean local charts. Here we factor the variation of the operators themselves and construct a mathematically consistent candidate function from the resulting coordinates.

## First native finding: activation geometry and operator geometry differ

On the same 4092 natural-training contexts, the shared mean accounts for **87.1% of normalized-input energy**, but only **5.5% of isotropic operator energy** and **6.6% of training-direction-weighted operator energy**. The remaining operator energy is context-dependent variation around that mean.

This is the exact mean/variance energy decomposition, not a fraction of the model understood or a causal faithfulness score. The task-direction metric averages a training direction covariance across every natural context; it does not preserve the empirical correlation between a particular context and its actual intervention direction. No held-out evaluation has been performed in this stage.

The distinction matters conceptually: a large common activation component can correspond to a relatively weak common local operator. Geometric similarity of hidden states does not, by itself, establish shared computation.

## The metric and fixed factorization

For the native bilinear write F(z)=D[(Lz)⊙(Rz)], J(z) is linear in z. We construct Q such that

(z−w)ᵀQ(z−w) = tr[(J(z)−J(w)) S (J(z)−J(w))ᵀ].

S=I measures full Jacobian Frobenius distance. The second choice uses the unnormalized second moment of 296 eligible source/final position changes from all 70 training pairs. Both directions of each pair are retained. The ordinary Euclidean metric Q=I is the activation-geometry control.

Metric PCA uses only the natural training contexts. It produces a shared mean operator plus context-dependent coefficients for a fixed operator basis. Ranks 4, 16 and 64 were fixed before fitting, and all are retained. Both operator metrics have numerical rank 1152 at the fixed relative cutoff 1e-10; this is not a native channel-rank or removability result.

At 64 context coordinates, the same-objective comparison is:

| Basis | Remaining centered isotropic operator variation | Remaining centered training-direction operator variation |
|---|---:|---:|
| Euclidean input PCA | 49.38% | 47.63% |
| Isotropic operator PCA | 48.16% | 46.17% |
| Training-direction operator PCA | 48.94% | 45.00% |

Each operator-aware basis improves its fitting objective, as expected from its construction. These are modest gains over Euclidean PCA; substantial centered operator variation remains. They do not establish a small set of semantic computational units. Including the shared mean, the rank64 matched operator models leave 45.51% and 42.01% of total operator energy unexplained, respectively.

## Why an operator basis is not automatically a featurizer

Projecting the context used to choose a Jacobian, while allowing unrestricted input-change directions, can produce a non-integrable operator field: there may be no function whose Jacobian it is. A simple example is F(z)=z0*z1. Keeping only z1 in the anchor gives a putative Jacobian [z1,0], whose cross-partials disagree.

We therefore separately implemented the consistent function

Fhat(z)=F(mu)+J(mu)(z−mu)+F(P(z−mu)),

where P is the learned linear context projection (with row/column convention handled in code). It retains a shared linear computation and a projected quadratic correction. Its exact finite change is

J(mu) delta + J(P(midpoint−mu)) P(delta).

Projecting both the context and change is crucial. The operator-fitting objective does not guarantee that this function will preserve native writes or downstream behavior. That is the next empirical test. It also does not yet imply an efficient implementation: the reference evaluator uses the original native weights, and the shared linear term remains full dimensional.

## Correctness and limits

Twenty-four random toy metric checks matched explicit Jacobian Gram matrices to relative error below 5.5e-16. A known low-dimensional toy with irrelevant high-variance input demonstrates recovery by operator PCA and failure by input PCA. A direction-limited metric also demonstrates a blind spot outside its selected directions. Positive channel duplication and signed opposing copies leave the function and metric unchanged, directly showing why operator geometry cannot identify implementation redundancy or cancellation.

Twelve independent symmetric-tensor checks verify the projected quadratic function, exact finite changes, full-rank recovery and finite differences (maximum relative discrepancy 2.21e-10).

The native audit independently reconstructed the training direction covariance, checked selected explicit Jacobian Gram entries, checked projection identities and replayed all 27 cross-metric comparisons with Torch. A separate direct uncentered-energy calculation verified the mean/variance decomposition. This is not an independent full metric-matrix or eigensolver reconstruction. No new native Transformer forwards were required.

Next: preflight all nine projected quadratic candidates plus the shared-linear and exact-native controls, then freeze and run held-out natural, local-change and downstream-causal comparisons. Keep both existing holdout cohorts, all ranks and all task families. Anchor-only operator predictions are diagnostics, not executable function replacements. No held-out rank selection, additional metric search, native removability claim or semantic-manifold claim is authorized by these setup results.
