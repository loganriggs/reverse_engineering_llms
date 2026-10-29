# Fixed-degree reuse: certified factors and an initialization trap

All81fits complete and independently audited. Degree is fixed at4 in3variables for every teacher and student. Models are products

f_A(x)=product_(i=1..g)(a_i dot x)^(4/g) / scale, g in{1,2,4}.

Teacher and initial student Gaussian RMS are1, with positive normalizers frozen during training. Three teacher seeds per family; three student families; symmetric, small-perturbation(.05Gaussian) and independent-factor initializations. PlainSGD1000steps atlr.03, each factor gradient divided by its multiplicity4/g. No clipping/momentum/regularization. Parameters3g: varying sharing necessarily changes the available direction count in this restricted family; this is not a parameter-budget-matched general neural-network comparison.

Loss is the exact Gaussian population MSE, evaluated using5-point Gauss-Hermite in each coordinate (125nodes). Squared quartic errors have degree at most8 in each coordinate, within this rule's exactness. This removes finite-data coverage and sample-size effects;125nodes are an integration rule, not125random training examples. Float64 implementation has numerical rounding, independently checked by Gaussian coefficient moments.

## Restricted reuse measure with an exact certificate

Within the grammar of products of homogeneous real linear forms, define R_product(f) as the minimum number of distinct projective linear factors (directions identified up to nonzero scaling). Unique factorization in R[x1,x2,x3] makes the factor multiset and multiplicities unique up to scaling/permutation. Consequently these teachers have R_product=1,2,4 respectively when their factors are nonproportional. Exact rational cross-minors of all saved teacher factors certify that condition. This is an exact complexity measure for a restricted grammar, not the minimal size of an arbitrary arithmetic circuit or a CP rank.

The four-factor student can represent all three teacher families. One-factor students cannot exactly represent genuine two-factor targets; one- and two-factor students cannot exactly represent generic four-factor targets. In addition, their even factor exponents and positive normalization force nonnegative outputs; generic products of four distinct real linear forms change sign across simple factor hyperplanes. These are representability obstructions, not certified numerical best-fit floors.

## A symmetry-invariant subspace

Suppose all student factors initially equal a and normalization is the same across group counts. Every model computes (a dot x)^4/scale. At this state, the derivative with respect to each stored factor is (4/g)*(a dot x)^3*x/scale. After dividing each gradient by4/g, all factors take the same update, independent of g. Induction proves identical effective trajectories for any deterministic sequence of data batches and differentiable output loss, with exact arithmetic. Identical factors never split spontaneously.

Thus enlarging the architecture from one to four independently trainable factors does not enlarge the functions actually visited from this symmetric initialization under this optimizer. Symmetric initialization plus matched functions can conceal an expressive advantage. This is a specific product-model symmetry: do not generalize it to all untied trees, whose occurrence roles can differ (the image models had nonzero disagreement gradients).

## Exact population MSE after1000steps

Medians over3teacher/initialization seeds; teacher energy1, so zero prediction has MSE1.

| Teacher distinct factors | Initialization | Student1 | Student2 | Student4 |
|---|---|---:|---:|---:|
| 1 | symmetric | 9.97584e-14 | 9.97584e-14 | 9.97584e-14 |
| 1 | perturbed | 1.41923e-11 | 3.54927e-07 | 2.4144e-05 |
| 1 | independent | 5.63983e-07 | 0.000295491 | 0.00528132 |
| 2 | symmetric | 0.0142241 | 0.0142241 | 0.0142241 |
| 2 | perturbed | 0.0142241 | 0.0141951 | 0.0129739 |
| 2 | independent | 0.0142241 | 0.00614989 | 0.00598685 |
| 4 | symmetric | 0.405501 | 0.405501 | 0.405501 |
| 4 | perturbed | 0.405506 | 0.404519 | 0.393745 |
| 4 | independent | 0.406818 | 0.372189 | 0.000453046 |

On generic four-factor targets, the four-factor student's median error is.4055 from symmetric initialization,.3937 with tiny perturbations, and.000453 with independent initialization. The capacity is available but symmetry and finite-time optimization determine whether training uses it. For one-factor targets, the simplest family learns well; overparameterized independent factors fit more slowly under this recipe. Correct-family two-factor fits retain error, so1000steps does not establish convergence. Independent initializations have matched RMS but different initial functions; this experiment intentionally varies symmetry and cannot attribute every difference to symmetry alone. A perturbation-size panel would isolate that transition more precisely.

## Validation and interpretation

Independent coefficient expansion plus analytic Gaussian moments reproduced all81final MSEs. Independently differentiated coefficients matched autograd gradients at all81initial states to maximum relativeerror1.742e-15. Symmetric models' final factor vectors agree across group counts within2.665e-15, and recorded loss trajectories agree. Allfits finite; none excluded. CPUonly, genericdispatch/core0, no native runtime crash this batch.

This establishes a fixed-degree reuse/learnability example and exact restricted complexity certificates. It does not establish a universal benefit of sharing, sample complexity, global minima, or physical circuit recovery. No recovery claim is inferred from low error; factor matching modulo scaling/permutation is the next independent check. Next discriminating controls: quantify factor recovery and initialization-perturbation dependence, then sample-size/noise tests with exact population evaluation retained. Keep full image scaling and theoretical complexity work separate but connected.
