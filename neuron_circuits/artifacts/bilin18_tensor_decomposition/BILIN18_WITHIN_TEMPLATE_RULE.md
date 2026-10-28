## H900 — A fixed within-template PCA rule beats shuffled controls, but new grammar contexts fail the competence gate

The previous turn made progress: H899 identified template-dominated PC1 and a causally useful PC2 in the frozen A/B output fit. H900 tests a predetermined extraction rule rather than selecting PC2 after seeing evaluation effects. Center discovery contextual writes separately within template A and B, then take the first residual PC. Decode with the same global discovery mean and an exactly compiled coefficient. The original number axis/origin and32 regular-noun discovery endpoints remain fixed.

Compare pooled PCA rank1/rank2, within-template rank1, and four rank1 controls using balanced random permutations of the template endpoint labels (seeds900–903). No number labels enter output fitting. Template labels enter discovery residualization only; no runtime template labels, gate, or per-template decoder are used. Fits are saved before live evaluation and hashes checked afterward. The rule is a labeled diagnostic, not an unsupervised featurizer.

### Evaluation eligibility and competence

Declare eight additional noun pairs: baker, actor, dancer, painter, guard, sailor, writer, visitor and their plurals. Use grammar tails “near the door” and “in the room,” plus matched word/item copying. Baker and painter pairs fail single-token copy-answer eligibility and are excluded before evaluation, leaving six pairs and48 contexts (24 grammar,24 copy). No correctness-based exclusions. These combinations are new to this investigation; global research/pretraining uniqueness is not claimed.

Both grammar templates retain100% binary is/are preference but fail the earlier full-vocabulary competence gate: exact-answer top1 is16.7% near the door and0% in the room. Copying passes at75% and100%. All evaluated contexts are retained. The grammatical margin results below are model-mechanism evidence in failed-gate contexts, not validated competent grammar-task transfer.

### Frozen-rule results

Intervention effect error is RMSE of exact versus approximate number-flip margin changes, measured relative to each condition's own native baseline. Native KL ratio compares replacement with removal of the same exact cross term.

|Rule|Grammar native KL/removal|Grammar flip-effect RMSE|Exact-flip to approximate-flip KL|Copy flip-effect RMSE|
|---|---:|---:|---:|---:|
|drop_cross|1.000|1.057|0.001932|0.108|
|pooled1|0.356|0.867|0.000704|0.053|
|pooled2|0.208|0.091|0.000256|0.056|
|within1|0.222|0.102|0.000276|0.057|
|shuffle0|0.356|0.867|0.000704|0.053|
|shuffle1|0.355|0.860|0.000699|0.053|
|shuffle2|0.356|0.861|0.000700|0.053|
|shuffle3|0.359|0.886|0.000717|0.053|

Within-template rank1 reduces pooled grammar flip-effect RMSE0.867→0.102 compared with pooled rank1, approaching pooled rank2 at0.091. Native KL improves0.000444→0.000277, and exact-flip approximation KL0.000704→0.000276. All four shuffled controls remain at effect RMSE0.860–0.886. This supports a role for meaningful grouping rather than arbitrary subtraction of group means. Four controls provide a diagnostic comparison, not an exhaustive permutation significance test.

The within-template direction has absolute cosine0.955 with the original PC2 and0.298 with PC1. Shuffled directions retain cosine0.9985–1.0000 with PC1. This independently specified rule recovers a direction similar to the earlier useful lower PC without using validation intervention outcomes to select it.

Copying does not show a corresponding intervention gain: within-template error0.057 versus pooled rank1 0.053, with a modest native KL/removal improvement0.818→0.792. Thus the method does not establish a general-purpose computational feature. Failed grammar competence further limits the semantic claim despite the strong improvement in margin-intervention fidelity.

### Next evidence needed

First evaluate these unchanged fits on the previously competent grammar contexts, explicitly labeling that evaluation as reused diagnostic data. In parallel scope for the next experiment, test whether unsupervised grouping of discovery activations recovers the same useful residual covariance; choose a fixed grouping rule using discovery only and retain failed groupings. No claim that labels have already been discovered automatically. Larger fresh competent-task panels and diverse non-number computations remain necessary before calling the resulting directions meaningful reusable manifolds.

### Verification

912 full-model forwards as planned for48 contexts and nine conditions at native/flip states plus baseline. Exact replacement is identity with zero logit difference. Compiled encoder relative error1.46e−15. Discovery covariance decomposition error1.55e−16 and frozen mean replay6.0e−16. Source/model/frozen discovery and newly saved output-fit hashes unchanged; all KL finite. Eligibility exclusions recorded. No natural or reserved-test evaluation; document pool remains26,112. Broad goal active.
