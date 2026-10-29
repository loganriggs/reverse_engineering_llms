# Releasing sharing after learning

200new independent-head fits:15learned shared solutions copied into independent heads, plus five exact-teacher independent initializations. Identical d8/h4data,512inputs, ten heads, noise.05,200TRFevaluations/head, no restarts. The release starts with exactly the same function as its corresponding shared model. Teacher starts are privileged diagnostics. Releasing expands164shared parameters to740independent parameters and adds computation; this is a causal continuation test, not a standalone optimizer leaderboard.

Transfer always uses ten normalized predicted source outputs and the same four held-out targets with128noisy labels/head. Whole-head holdouts remain untouched by source fitting. All per-head empirical losses decrease or remain equal after continuation, verified independently.

| Method | Median teacher training MSE | Median source population MSE | Median transfer MSE | Median population transfer floor |
|---|---:|---:|---:|---:|
| shared | 0.00248269 | 0.00020528 | 0.000851505 | 7.52507e-05 |
| separate_width | 0.00953396 | 0.0313684 | 0.109226 | 0.086619 |
| released | 0.00222953 | 0.00114873 | 0.0343603 | 0.0232988 |
| teacher | 0.00217358 | 0.000995241 | 0.0343603 | 0.0232988 |

Each teacher aggregate averages all three original initialization outcomes where applicable. Teacher initialization is run once, not triplicated.

| Teacher | Shared transfer | Released transfer | Teacher-start independent transfer |
|---|---:|---:|---:|
| 225001 | 0.000462128 | 0.0343603 | 0.0343603 |
| 225002 | 0.000597858 | 0.101428 | 0.101428 |
| 225003 | 0.000851505 | 0.0496789 | 0.049679 |
| 225004 | 0.0122857 | 0.0209211 | 0.0129291 |
| 225005 | 0.00492277 | 0.00669109 | 0.00586085 |

## Main inference

Releasing sharing reduces training error yet worsens transfer for all five teacher aggregates. Independent models started at the exact teacher exhibit similarly degraded transfer. Thus random-start basin failure does not explain the entire shared-versus-independent gap. The sharing constraint also protects the learned source-function span under finite noisy observations in this experiment. This conclusion is specific to output-span adaptation; an independent model may contain useful information accessible through a different feature extraction procedure.

## Why small source errors can damage transfer

The ten normalized teacher outputs span ten hidden products, but their smallest directions are weak: condition numbers range60.8–746.8. Teacher-start independent fits have source population MSE onlyabout.0007–.0011, yet their minimum principal cosines with the true output span range.072–.801. The13shared fits that passed the component-recovery threshold retain cosines above.999; two failed shared starts are retained and do not meet that description. These are teacher-informed diagnostics, not training supervision. Independent orthogonal-projection replay confirms the transfer floors.

For a full-column-rank teacher output matrix T and fitted matrix F=T+E with the same rank, let P_F project onto col(F). Since (I−P_F)F=0, (I−P_F)T=−(I−P_F)E. Writing an orthonormal teacher basis as T times its inverse singular factors gives sin(theta_max)<=min(1,||E||_2/sigma_min(T)). The checker verifies this bound on every diagnosed fit. It is often loose; a large bound does not prove instability.

Shared structure constrains the form of these errors. Write the source coefficient matrix C=K Wᵀ, with K the common hidden-product functions and invertible square W. Its first-order perturbation is deltaC=deltaK Wᵀ+K deltaWᵀ. Projecting off col(K) and undoing W gives (I−P_K)deltaC W⁻ᵀ=(I−P_K)deltaK: the readout perturbation is in-span, and the inverse-W factor cancels for coherent dictionary changes. Arbitrary independently fitted head errors need not have this structure and can be amplified by W⁻ᵀ. This elementary local identity motivates the observed mechanism; it is not a proof that every shared estimator is statistically better.

## Verification and scope

All20groups/200headfits independently audited: initial-function copying, seeded-data provenance, initial/final autograd Jacobians,400fresh-input polynomial checks, every initial/final empirical loss, exact source/transfer Gaussian errors, normalization and adaptation normal equations. Maximum relative Jacobian discrepancy7.45e-16. Source snapshots, checkpoints, solver outcomes and all noise realizations retained. No failure selection or adaptive retries.

The earlier13/15explicit optimization-failure certificates remain valid. This experiment adds a second effect: removing the shared constraint can reduce empirical loss while losing transferable structure. Future tests should distinguish coherent dictionary errors from independent head errors or compare alternative extraction of the independent hidden features, rather than assigning the whole gap to optimization alone.
