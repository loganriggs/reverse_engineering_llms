

## H261 — Explicit response training learns supervised sources but harms held-out sources

H260 made progress: natural-state predictors failed causal-transfer thresholds.
H261 tests whether explicit response supervision suffices with the SAME H259 feature
maps and readout parameter counts. Earlier L8 direct-objective work showed limited
source transfer on four output coordinates; this experiment concerns the full L17
compiled-program residual and downstream mediation.

**Frozen design.** Affine1152 and quadratic1152 maps, PCA/centering/scales fixed
from H259. Train on first768 discovery documents; validate last256. Response targets
are full-deletion missing-write changes for heads0/2/4/6/8 only. Heads1/3/5/7 never
enter response fitting or selection. Natural states contain all heads, so holdout
means response-supervision holdout, not removal of those heads from the network.
The predictor receives no head ID. Joint loss = natural normalized MSE + beta times
response normalized MSE, beta0/1/10, with training energy denominators. Response
rows have zero intercept; constant errors cannot masquerade as response learning.
Closed-form double-precision ridge .01/.1/1, selected per beta using corresponding
validation objective. beta0 uses natural validation only. No refit. beta1 primary;
beta10 secondary, not a post-test winning choice. All candidates retained.

Full and10%-edit responses on128 H259 documents are evaluated using H260 common-
native-baseline mediation and float64 KL. Same-head norm/angle controls probe
transfer beyond actual source directions. All coefficients frozen before evaluation;
no test-native residual enters a predictor. Evaluation documents are reused, not
fresh, although none are discovery documents.

| Quadratic response weight | Actual supervised fidelity, full edit | Actual held-out fidelity | Natural KL recovered |
|---|---:|---:|---:|
|0|4.72%|5.89%|42.17%|
|1|38.04%|−.85%|44.90%|
|10|40.43%|−7.53%|37.39%|

**Supervised-learning gate PASSES.** beta1-minus0 supervised fidelity gain33.32
points[30.32,36.24] at full scale,32.73[30.13,35.35] at small scale. Conditional KL
improves40.29%[34.44,46.59] and42.14%[35.77,48.68], respectively, versus beta0.

**Held-out-transfer and behavioral-transfer gates FAIL, with measured harm.**
Held-out fidelity gain is−6.74 points[−8.03,−5.54] at full scale and−6.55
[−7.84,−5.35] at small scale. Held-out conditional KL worsens4.00%[1.66,6.37]
and4.36%[2.06,6.79]. This is stronger than merely failing a10-point threshold:
paired document intervals exclude zero in the harmful direction. beta10 increases
supervised fidelity slightly but harms held-out fidelity by13.42 points and worsens
held-out KL7.86% at full scale. Affine shows the same tradeoff: full supervised
fidelity .72→19.88→24.43%, held-out−.47→−12.17→−22.32% for beta0/1/10.

**Direction-control result.** Quadratic full-edit angle-control fidelity on the
supervised heads is4.04→7.27→5.55%; on held-out heads5.53→.74→−4.45%. Thus the
38% actual supervised fidelity is not generic coverage of each supervised head's
whole output subspace. Small-edit angle controls show the same qualitative pattern.
This suggests that actual source/context joint statistics matter, but does not
establish a semantic binding mechanism or disentangle direction coverage from
feature misspecification.

**Natural-state tradeoff.** Quadratic beta1 mildly improves native KL recovery
42.17→44.90%, so its failed response transfer is not explained by global natural-
state collapse. beta10 falls to37.39%. Corresponding top1 agreements87.30/87.30/
86.52%, CE increases .04371/.03804/.05066. Affine KL recovery37.35/35.96/21.19%.
These beta0 models use natural-vector validation, unlike H259's KL-selected models;
quadratic beta0 selects ridge1 rather than H259 energy ridge.1. Comparisons therefore
use H261's own matched beta0, not a substituted earlier baseline.

All fitted normal-equation relative residuals<=2.94e-15 before float32 deployment.
One initial training process segfaulted before producing a model; unchanged retry
completed. The dependent evaluator initially lacked its file, then completed after
successful training. Neither failure produced an experimental result. Six directly
relevant tests pass (joint fit/intercept, causal patch, feature/solver and angle
controls). Figure visually inspected. All H261 runs complete; original weights
unchanged, all outcomes retained.

**Interpretation.** The feature map can encode useful supervised responses, so an
absolute inability to learn any response is weakened. Yet changing the objective
does not produce a shared transferable rule: it trades improvements on trained
sources for worse untrained sources, even without a source-ID input. This supports
source/context-limited fitting and keeps representation capacity/directional
coverage as competing explanations. It does not imply heads themselves are the
fundamental units, that every possible loss fails, or that features are semantic.

**Next information-gain comparisons.** First reverse the supervised/held-out head
split to test whether this asymmetry belongs to particular heads. Then match the
number and norms of training edits while replacing actual edits with same-head
angle controls or isotropic directions. Keep feature maps/readout counts fixed.
Broader transfer from direction-diverse supervision would support coverage as the
bottleneck; persistent tradeoffs under diverse supervision would implicate the
fixed function class or representation. Use validation from supervised conditions
only, report both splits and all controls, and do not train/test-select an overall
winner. Broad research goal remains active.
