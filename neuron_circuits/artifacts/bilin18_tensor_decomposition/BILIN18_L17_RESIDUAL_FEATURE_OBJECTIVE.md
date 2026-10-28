

## H259 — Both quadratic features and state-specific sensitivity training help

H258 made progress: a non-oracle affine predictor repairs part of the missing
write, leaving a substantial coefficient gap. H259 separates two explanations
with a fixed feature/objective factorial and new-document evaluation. Earlier L8
objective experiments targeted four source-response coordinates; this targets the
full1152-dimensional residual of the executable L17 compiled program.

**Design fixed before outcomes.** Train-only PCA of normalized pre-MLP activations.
Affine1152 uses all1152 linear PCs. Quadratic1152 uses624 linear PCs plus528 upper-
triangle products of the first32 PCs, off-diagonals scaled sqrt2. Affine624 is an
additional control. Features are centered and standardized on training only.
The core1152 models have equal fitted coefficient counts, but different function
classes: quadratic replaces528 lower linear PCs with interactions. The624 control
isolates gains from adding interactions without that replacement. No claim of
exhausting useful nonlinear representations or matching exact total execution cost.

Cross each family with ordinary residual MSE and a state-specific sensitivity
metric: M_i=(I + d/mean||g||² * mean_k g_ik g_ik')/2. Each g is an exact native
predicted-log-probability score gradient;16 categorical samples per training site,
fixed seed. The average trace matches the Euclidean metric. This is an energy-plus-
sampled-local-Fisher surrogate, NOT exact finite KL optimization or observed-label
training. State-specific weighting is essential; a single global output weighting
would not test the same explanation.

First768 discovery documents train; last256 validate. Fixed ridge .01/.1/1 selected
separately by validation exact native KL. No refit: all trained maps and selections
are frozen before model evaluation on128 new documents. New complete-document
hashes exclude1344 prior discovery/test documents; exclusion audit passes. Cached
corpus only, not a training holdout or cross-distribution replication. Test-native
error is used only for scoring and oracle control; corrections depend solely on
pre-MLP activations and frozen weights. No per-state inference chart is needed.

| Frozen selected predictor | KL recovery | CE increase | Native top1 agreement | Residual energy recovered |
|---|---:|---:|---:|---:|
|Affine1152, energy|37.35%|.04622|86.13%|29.69%|
|Affine1152, sensitivity|45.31%|.03804|87.11%|23.08%|
|Affine624, energy|39.79%|.04455|86.52%|35.94%|
|Affine624, sensitivity|47.20%|.03386|88.48%|34.56%|
|Quadratic1152, energy|45.58%|.03889|86.33%|39.24%|
|Quadratic1152, sensitivity|52.62%|.03094|89.45%|36.39%|

Compiled baseline KL .10135, CE increase .08710, top1 agreement82.81%. Full oracle
replays exactly. Core selected ridge strengths: affine1 under both objectives;
quadratic .1 energy and1 sensitivity. All18 candidates and validation scores saved.
H258 used different feature scaling, training/refit protocol and test documents;
its45% affine result is not a matched baseline for this table.

**Discriminating contrasts.** Quadratic-minus-affine1152 KL recovery gain is8.23
points[5.33,11.10] under energy and7.31[4.01,10.70] under sensitivity. Adding
interactions to affine624 gives5.80[3.78,7.87] and5.42[3.43,7.59]. Sensitivity-minus-
energy gain is7.95[4.94,10.94] points for affine1152 and7.04[4.02,10.52] for quadratic.
Intervals are paired document-bootstrap95%, conditioning on fitted models and
fixed gradient samples. Both locked >=5-point benefit gates PASS (point estimates
plus interval lower bound>0). Behavioral-repair gate FAILS: no selected core model
meets80% KL recovery, CE<=.02 and98% top1. Neither success is a semantic-unit claim.

**Numerical audit retained.** Original300-iteration CG run left three nonselected
sensitivity fits above the numerical convergence check. All selected models already
converged. A separately saved post-outcome1200-iteration audit brings maximum exact
normal-equation residual to1.16e-5 (all below the5e-5 verification threshold).
Every selected name and selected test score is unchanged exactly. Original results
are preserved under the original stem; audit files use _solver_audit. This was a
numerical check, not a new independent replication or retrospective gate change.

**Costs and interpretation.** Both core readouts fit(1152+1)*1152=1,328,256 scalars;
affine624 fits720,000. Quadratic inference also needs624 PCA projections and528
products; preprocessing storage and operation counts differ. Affine preprocessing
can be folded into one dense affine map, whereas quadratic terms need additional
basis/product coefficients. Therefore equal readout coefficient count does not
prove equal total storage/FLOPs. Actual compiled deployment remains to be measured.
The data support both missing interactions and an imperfect fitting objective;
neither alone explains the coefficient gap. These are continuous PCA products,
not identified semantic primitives. Lower energy accuracy can still improve KL.

Five directly relevant tests pass: conditional-metric CG versus explicit dense
normal equations, symmetric-quadratic feature norm, affine fit equations, exact
score gradients and low-rank reconstruction. Full project suite not rerun. Plot
inspected. H259 and convergence audit complete; native model unchanged.

**Next discriminator: causal transfer of frozen predictors.** Without retraining,
remove/attenuate actual L17 attention-head writes at the pre-MLP residual. Recompute
RMS and compare the predicted CHANGE in missing write against the native change;
then evaluate downstream response mediation using a common native baseline so
natural-state offsets do not dominate. Test small and full edits, all9 heads, and
norm/angle-matched controls where appropriate. Compare frozen affine, quadratic,
energy and sensitivity variants from this factorial. Natural-state prediction
need not transfer to source interventions; transfer would connect the learned
function to exercised computation, while failure would expose correlations or
missing response directions. Read older finite/source-mediation protocols first.
Do not call a high natural-state score a recovered causal mechanism. Goal active.
