

## H267 — Exact tensor partners beat random partners but barely beat within-subspace products

H266 made progress: residual-sensitive input directions helped modestly, while
source-variance directions hurt. H267 tests whether mixed interactions with a
separate partner subspace improve the remaining residual program. Earlier H248
compiled partner machinery motivated this test, but the object here is the
NATIVE MINUS ALREADY-COMPILED residual quadratic, not the full original MLP.

Freeze core16 from H266's residual-Jacobian32 basis. For each core direction c,
construct exact full-output M(c) with M(c)v=B_native(c,v)−B_compiled(c,v), using the
homogeneous quadratic part only (offset/linear/bias terms cancel). Accumulate
sum_c M(c)'M(c), project input onto the orthogonal complement of core16, and take
top24 eigenvectors. This maximizes retained restricted contraction Frobenius energy
for a24D partner span; it need not maximize exercised response fidelity. Compare
three fixed Gaussian QR random-complement24 controls. All controls share the core.

Common624 linear PCs. Partner models use136 within-core products plus384 mixed
core16×partner24 products=520 nonlinear features, no partner-partner terms. Within32
baseline retains H266's528 products; core-only baseline has136. Tensor and random
readout counts match exactly. Relative to within32, mixed models use8 fewer products
but up to8 additional input reads; no equal-total-compute claim. Coefficients fit
all-head response targets on768 discovery docs; fixed256-doc validation chooses
ridge from0/1e-6/1e-4/.01/.1/1. All6 models select.1. All36 fits retained, no refit,
no test basis selection. Same128 test docs reused; no semantic labels or head-ID
input. Predictor inference uses no native residual target.

| Model | Selected train fidelity | Validation | Actual test full edit | Angle-control test |
|---|---:|---:|---:|---:|
|Core16 only|43.92%|36.18%|35.01%|−2.01%|
|Random partner0|49.91%|38.94%|37.61%|2.64%|
|Random partner1|50.11%|39.19%|38.06%|2.65%|
|Random partner2|49.78%|38.88%|37.65%|1.94%|
|Within32|53.02%|43.08%|41.65%|6.89%|
|Tensor partners|53.02%|43.65%|42.32%|9.94%|

Tensor-minus-random full-edit gains4.71[4.40,5.06],4.27[3.96,4.60],4.68[4.35,4.99]
percentage points (paired document-bootstrap95%). Small-edit gains4.71/4.29/4.71.
Partner-specificity gate FAILS: required>=5 points against EACH random control at
BOTH scales; observed effects are positive but smaller. Mixed-advantage gate FAILS:
tensor-minus-within32 is only .67 points[.33,.96] full scale and .44[.04,.78] small,
not required5. Half-response gate FAILS. Tensor improves core-only by7.31 points
[6.87,7.79], but this also adds384 fitted product features, so not a pure selection
advantage. No thresholds are retrospectively relaxed.

**Weight geometry versus exercised response.** Tensor-selected partners capture
18.56% of the core-to-complement residual contraction energy; random controls capture
2.06/2.09/2.11%. That roughly9x weight-energy advantage gives only4–5 points of
actual response advantage. These energy percentages are NOT fractions of the entire
residual tensor, native model computation, or data response. They condition on the
chosen16 core directions and Euclidean full-output weighting. The gap cautions
against treating spectral concentration alone as an explanation of execution.

**Costs/audits.** Readout scalars including intercept: core-only876,672; tensor/random
1,319,040; within321,328,256. Input projections, feature normalization and the
unchanged original compiler are additional costs. Tensor basis construction contracts
16 full1152x1152 residual bilinear matrices before eigendecomposition; no runtime
basis construction is needed. Maximum core/partner overlap6.83e-8, orthogonality
error4.77e-7, fitted normal-equation residual1.01e-14. Exact bilinear contraction
passes a polynomial-polarization test that cancels arbitrary offset/linear terms.
Five relevant tests pass, full project suite not rerun. Figure inspected. All H267
runs complete; native model and original compiler unchanged. Angle-control seed
changed for this experiment; compare variants within this run, not absolute angle
numbers across previous reports.

**Interpretation.** Specific mixed couplings exist and are more useful than random
partner couplings at matched readout size, but this selection does not substantially
outperform within-subspace products or provide a complete response representation.
Possible remaining errors include core interactions with omitted partners, entirely
outside-core quadratic terms, and coefficient estimation/linear-read restrictions.
The current score cannot distinguish these. There is no discovered semantic unit
or evidence that the residual is fundamentally a core16-by-partner24 computation.

**Next discriminator: exact response-term accounting before another fit.** Center the
known residual polynomial at the fixed discovery mean. Decompose source responses
into mean-linear, within-core, core–retained-partner, core–omitted-partner, and
outside-core quadratic changes. Verify exact replay and report signed overlaps as
well as energies because terms can cancel. Compare structural truncations with
full mean-linear versus the existing624-linear restriction. Exact tensor coefficients
make these diagnostics, not cheap learned programs; a structural truncation is NOT
an upper bound on the best fitted readout, which can exploit correlations among
omitted terms. This locates the missing computation before spending more parameters
on another family. Broad research goal remains active.
