# Preregistration: predict the local suffix route from state

The mask surface has useful pairwise selection structure, but its coefficients
vary by site. This test learns a frozen state-conditioned pairwise router.

For every candidate mask, use its nine inclusion bits and 36 pair products. A
router predicts finite fidelity with a linear mask model plus bilinear products
between mask features and one of three site representations:

1. eight PCs of the normalized Layer-8 state;
2. eight PCs of the 27 singleton-utility, leaveout-necessity, and singleton-
   energy diagnostics;
3. both feature sets.

Training uses only the first FineWeb site for each of 16 tensor queries. Ridge
strength is selected by four-fold held-out-query routing fidelity, not rowwise
mask prediction. The router is frozen and evaluated on the second FineWeb site
for each query and both WikiText sites. PCA is fitted on training sites only.

A router passes at median exact finite fidelity >=0.95 and 10th percentile
>=0.80 on both evaluation corpora. The combined router must additionally gain
0.05 at the 10th percentile over each component somewhere. All models must
exactly reproduce their own saved predictions after serialization, and training
fit is reported to distinguish optimization failure from transfer failure.

This is a causal-data-trained interpreter: training targets come from finite
interventions. Success would establish a predictable address rule, not a purely
weight-only derivation. Failure applies only to this low-rank bilinear router.
