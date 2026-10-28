

## H304 — Output spectrum of the missed rank64 cross response

Frozen H300 refined64, fixed core96. For each of18 head/payload groups collect the uncentered second moment of exact cross response minus projected-partner prediction. First768 discovery documents train output eigenvectors; next256 evaluate them. Same payload seed. Pooled selector sums raw moments; equal-group selector sums each moment divided by its own missing-energy trace. Fixed ranks8/16/32/64/128/256 plus a seeded random orthonormal output basis. No centering: explained energy includes any common mean response. No fresh documents, further input fitting, or checkpoint selection.

| Selector | Output rank | Training equal-group capture | Validation equal-group capture | Worst validation group |
|---|---:|---:|---:|---:|
|pooled|8|2.66%|2.68%|2.18%|
|pooled|16|4.86%|4.77%|4.03%|
|pooled|32|8.65%|8.21%|7.42%|
|pooled|64|15.42%|14.40%|13.13%|
|pooled|128|27.10%|25.06%|23.41%|
|pooled|256|45.79%|42.77%|40.64%|
|equal_group|8|3.13%|2.64%|2.17%|
|equal_group|16|5.69%|4.87%|4.20%|
|equal_group|32|10.15%|8.51%|7.41%|
|equal_group|64|17.76%|14.92%|13.18%|
|equal_group|128|30.21%|25.93%|23.77%|
|equal_group|256|49.21%|43.89%|41.85%|
|random|8|0.70%|0.70%|0.66%|
|random|16|1.39%|1.41%|1.33%|
|random|32|2.78%|2.80%|2.72%|
|random|64|5.56%|5.59%|5.47%|
|random|128|11.11%|11.15%|10.94%|
|random|256|22.22%|22.23%|22.06%|

Locked gates: compact32: FAIL, compact128: FAIL, specificity32: FAIL. Compact32/128 requires>=80% missing energy captured in EVERY validation group, using equal-group-selected directions. Specificity32 requires>=10pp advantage over random in EVERY group.

The equal-group basis captures only8.51% validation missing energy at rank32,25.93% at128 and43.89% at256; worst groups capture7.41%,23.77% and41.85%. Thus a small shared output bottleneck does not explain the missed cross response. This applies to the residual after existing approximations, not the original complete MLP write. The next discriminating comparison is shared versus per-group output spectra: broad pooled outputs could reflect either broad within-group writes or a union of different compact head-specific spaces.

**What this measures.** These are oracle projections of already-computed missing responses. A shared output subspace can be compact even if the input-to-output map requires many partner directions. Output capture therefore does not establish an executable predictor, causal semantic units, or behavioral sufficiency. It does identify whether further input-side fitting is trying to predict a few output coordinates or a broad write space. The statistic concerns missing cross response only, not the full remaining model error.

**Numerical checks.** Uncentered covariance trace matches directly accumulated residual energy to relative<1e-10. Saved output bases are orthonormal at1e-10 tolerance. CPU recomputation from saved moments reproduces per-group validation captures at ranks32/128 within1e-10. Native responses use float32; moments and eigendecomposition use float64. These checks validate the projection accounting, not native-model approximation accuracy.

**Next discriminator.** If compact output coverage transfers across groups, project the exact bilinear residual coefficient tensor into that output span and inspect its input coupling spectra. Count the cost of predicting those coordinates using shared versus output-specific low-rank factors. Validate the resulting factorization on response differences before any behavioral claim. If output coverage remains broad, compare direction sharing across groups and mean-centered moments before choosing a more expensive representation. This is a change of representation diagnostic, not another optimizer sweep.

All jobs terminal; original model/compiler/base unchanged. Existing validation only; broad research goal active.
