

## H258 — Non-oracle residual prediction finds signal but leaves a coefficient gap

H257 made progress by isolating readout-metric dependence. H258 returns to an
executable internal computation: predict the compact quadratic program's missing
write from normalized pre-MLP activation x, without test-native residual access.
Earlier H246 linear models targeted source-response differences; this experiment
fits the natural-state residual of the explicit H248 compiled program.

Train centered affine ridge predictors on first768 of the1024 discovery documents;
validate on last256. Fixed strengths .001/.01/.1/1, lambda=strength*trace(Xc'Xc)/1152.
Choose full-vector and chart16 variants separately by validation exact native KL.
Both select .1. Refit all four on1024 discovery and save selected names/maps BEFORE
collecting test states. Evaluate all variants on H256128 documents, explicitly now
reused after earlier diagnostics; hyperparameter selection remains discovery-only.
Chart64 uses the chart16-selected coefficient model. Original support256/T=1.

No test-native errors enter either affine predictor or chart construction. The
native model is used for evaluation targets and explicit oracle controls only.
Constant correction is mean discovery error; charted constant uses the same local
projection as the candidate. Coefficients are no longer oracle in predicted rows.

| Correction | KL error recovered | CE increase vs native | Native top1 agreement | Error energy recovered |
|---|---:|---:|---:|---:|
|Compiled only|0%|.10995|82.62%|0%|
|Constant|.78%|.10954|83.59%|.46%|
|Constant into local16|.82%|.10841|83.59%|.003%|
|Selected affine, full|44.80%|.04718|87.30%|35.58%|
|Selected affine into local16|36.75%|.06092|86.91%|1.01%|
|Selected affine into local64|40.55%|.05189|86.72%|2.47%|
|Oracle local16|81.21%|.00801|100%|2.28%|
|Oracle local64|90.17%|.00696|99.80%|6.36%|

Predictive-signal gate PASSES: selected chart16 recovers>=20% KL and beats constant
chart16 by−.03479 mean KL, paired document-bootstrap95%[−.04496,−.02572].
Oracle-gap-closed gate FAILS: predicted chart16 achieves45.26% of oracle chart16's
KL reduction, below80%. Non-oracle behavioral-repair gate FAILS for both selected
variants. Oracle16 descriptively satisfies all three repair thresholds on this
panel now that CE/top1 have been measured; this does not turn the non-oracle gate
into a success or erase H253's failure on its earlier panel. Full oracle is exact.

Every fixed regularization result is retained. Full-vector recovery ranges40.7–44.8%;
chart16 ranges33.0–36.8%. Strongest ridge recovers38.6% energy but40.7% KL, versus
selected35.6% energy/44.8% KL: Euclidean accuracy and behavior remain distinct.
Full-vector prediction outperforms projecting that prediction into local charts.
This is observational evidence about these fits, not proof charts can never help.

**Cost.** One affine model adds1152x1152+1152=1,328,256 float32 scalars (5,313,024
bytes),1152 input coordinates and a dense matrix multiply. With the504,224-scalar
compiled program, total1,832,480 scalars (~11.51% of native15,926,400). This is a
parameter count, not measured wall-time speedup. The full-vector variant requires
no chart or extra vocabulary evaluation. Charted variants additionally require
full vocabulary probabilities, score gradients and per-state SVD, so they are
expensive diagnostics despite16 projected output coordinates. Program unchanged.

**Interpretation.** Missing relevant values are partly predictable from available
activations using a simple affine function. Yet much of the local oracle advantage
is unavailable to that predictor. A low-dimensional behavioral interface does not
imply a comparably simple computation of its coefficients. This narrows T08/T09/T10:
useful directions, coefficient predictability, execution cost and semantic meaning
remain distinct. Failure is restricted to this affine class/data/objective, not all
nonlinear encoders or possible internal units. Four relevant tests pass, including
ridge normal equations/intercept, projection, score gradients and reconstruction.
All H258 runs complete; native weights untouched.

**Next informative discriminator.** Test whether the affine coefficient gap comes
from missing quadratic interactions or the Euclidean training objective. Use a
matched-budget factorial: affine versus fixed quadratic input features, ordinary
residual loss versus downstream-sensitive loss, train/validate only on discovery,
and freeze before a new-document evaluation. Count feature construction and solve
cost. Do not infer nonlinearity from the affine failure alone. Inspect existing
quadratic compiler forms/earlier objective experiments first to avoid a redundant
comparison. A useful result must separate these two explanations rather than just
raise one score. Broad research goal remains active.
