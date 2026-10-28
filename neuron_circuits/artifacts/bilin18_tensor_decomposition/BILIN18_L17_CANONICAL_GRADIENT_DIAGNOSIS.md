## H467 — Small-step diagnosis of data-canonical objectives

Keep the H466 empirical64-dimensional bilinear coordinates, rank16 map, training256/validation128 split and balanced-SVD initialization fixed. Compute full-training and validation gradients for KL and observed-token CE. Normalize each negative training gradient so its first-order effective-map change has Frobenius norm equal to the baseline map norm. Sweep fixed relative fractions0,1e-5,3e-5,1e-4,3e-4,.001,.003,.01,.03,.1. Evaluate train/validation write MSE, KL and CE; do not evaluate test metrics. This is a reused-data diagnostic without confirmatory gates or new corpus documents.

Both objectives have training directions that locally improve validation. KL training/validation gradient cosine is0.09739; CE cosine0.03048. These are weak positive alignments, not evidence of robust global agreement. Validation directional derivatives along the KL training direction are−0.368791 for KL and−0.413797 for CE. Along the CE training direction, they are−0.102834 for KL and−0.321311 for CE. Thus both directions initially improve both validation behavioral objectives; there is no local antagonism at this point. Write accuracy is optimized by the initial reduced-rank solution on training and can worsen under these behavioral updates.

Symmetric finite differences at fraction.001 agree with analytic own-objective derivatives: train KL−1.869223 versus−1.868841; validation KL−.368874 versus−.368791; train CE−8.130779 versus−8.130919; validation CE−.321533 versus−.321311. These are numerical directional checks, not estimates of population uncertainty.

| Update | Relative fraction | Train KL | Validation KL | Train CE | Validation CE | Validation write MSE |
|---|---:|---:|---:|---:|---:|---:|
|Initial write optimum|0|0.089266|0.145440|3.076032|3.267789|0.152321|
|KL direction, own-validation selection|0.01|0.075540|0.142932|3.056447|3.264818|0.152337|
|CE direction, own-validation selection|0.03|0.093685|0.148260|2.854113|3.264026|0.152988|

The validation-selected KL fraction.01 improves validation KL from.145440 to.142932 and CE from3.267789 to3.264818. The selected CE fraction.03 improves validation CE to3.264026 while worsening KL to.148260 and write MSE from.152321 to.152988. This is the first objective-dependent tradeoff in this canonical-form pilot. It is a validation-selected point, not a confirmed test improvement. At the smaller CE fraction.01 both validation KL and CE improve, demonstrating that antagonism depends on step size rather than being unavoidable locally.

Both directions overshoot at fraction.1: validation KL/CE become.218762/3.335236 for the KL direction, and.246458/3.346675 for the CE direction. This shows that useful local validation updates exist and overly large moves can harm validation. It does not establish that H466 Adam failed solely because of its learning rate: stochasticity, accumulated path drift, regularization and differing optimizer geometry remain alternatives. The initial map's replay and normalized first-order step audits pass. Balanced factor coordinates are held fixed; steepest descent depends on this chosen parameterization even though the represented function can be canonicalized afterward.

The selected KL.01 and CE.03 updates are frozen in the selected-steps artifact, with exact saved delta-factor directions. Next evaluate those fixed alternatives and the write baseline on the H466 test split, reporting write accuracy, KL, observed-token CE, and intervention-delta prediction separately. This is new alternative evaluation on a historically viewed test panel, not independent corpus confirmation. Use future fresh documents for any resulting population claim. After that, test repeated small-step or regularized optimization if required; no conclusion about global objective optima is warranted.

All12992 tracked corpus documents are unchanged. The broader objective remains open: the experiment diagnoses how different notions of behavioral equivalence choose approximations; it does not yet identify a universal feature ontology.
