## H891 — A compact conditional write approximates the L12 number–context intervention

The L12 interaction vector is more reusable than a fixed mean, but needs some contextual variation. One discovery-derived output direction plus the mean reproduces individual cross-term intervention effects on new agreement nouns with about7–18% error. A random seven-dimensional subspace barely improves the mean. The approximation can be compiled into two output directions controlled by a number-coordinate change and one linear context coefficient. This is a candidate compact conditional operation, not yet a replacement validated on native model behavior.

### Target and discovery fit

The target is H890's midpoint-centered cross-term intervention Δcross=δ*C(w), not the full coordinate flip or the entire MLP. C(w) is the context-dependent bilinear interaction vector and δ is the frozen number-coordinate change. Use the original eight discovery endpoints (dog(s), cat(s), book(s), tree(s)), separately for each original trained axis/template and task/copy family. Fit a mean and centered PCA basis of C(w), with ranks1,2,4,7 fixed before these approximation outcomes. Include one random rank-7 basis, wrong-family mean and pooled-family mean controls.

Evaluate all120 H887 eligible new noun contexts, both frozen axes and all original selected templates. The newer nouns are excluded from fitting, but were already used to localize L12 and inspect its interaction. This is posthoc, estimation-disjoint compression evidence, not untouched prospective confirmation. Each fit has only eight points; rank7 exactly reconstructs centered discovery data by construction.

### Agreement intervention fidelity

Relative error compares individual expected-number margin changes under the approximation with the exact cross-only intervention, before averaging examples. KL compares the exact edited output distribution with the approximated edited distribution. Its ratio uses KL(exact edited || native unedited) as denominator; it is not a fraction of semantics explained.

|Cohort|Axis|Mean margin-effect error|Rank1 error|Rank7 error|Random7 error|Rank1 KL ratio|Rank7 KL ratio|
|---|---|---:|---:|---:|---:|---:|---:|
|irregular|A|0.224|0.073|0.035|0.220|0.033|0.023|
|irregular|B|0.567|0.084|0.040|0.562|0.060|0.052|
|new_regular|A|0.402|0.072|0.034|0.395|0.060|0.049|
|new_regular|B|1.002|0.175|0.095|0.995|0.190|0.178|

Rank1 improves strongly over the mean in every agreement cohort/axis cell. Additional ranks improve selected-margin fidelity, but the weakest cell (new regular, axisB) retains a vocabulary KL ratio of0.178 even at rank7. A low margin error therefore does not imply complete output-distribution fidelity. Polarity-separated errors and absolute margin-effect errors are retained in the analysis.

Copying does not show the same successful rank1 transfer. Its individual cross-intervention margin errors are about25–60% at rank1 and26–69% at rank7, with KL ratios still0.226–0.527 at rank7. This does not resolve the broader copying coverage problem or establish a universal number-axis operation across tasks.

### Compiled conditional operation

Write the rank1 approximation as C_hat(w)=mu+beta(w)*v, where v is the discovery output principal direction. Because C(w) is linear in w, beta(w)=v·(C(w)−mu) is itself an affine function of the normalized native input u after accounting for the frozen coordinate origin:

beta(u)=encoder·u+offset,

Δcross_hat=δ*mu + δ*beta(u)*v.

The encoder is derived analytically from the model's Left/Right/Down weights and v. It does not require computing the full C(w), and no new parameters are fitted to the transfer examples. The saved compiled units contain the frozen number axis/origin, discovery number centroids, context encoder/offset, and the two output vectors. There are separate axis/family units; this is not a single universal task-blind unit.

This representation has a number-dependent write and a product of number change with a context coefficient. It is concrete evidence for a compact conditional computation within the tested intervention. The experiment does not establish the semantic interpretation of beta, its stability across models, or that the rest of the layer can be omitted. It also does not show that an unsupervised featurizer would discover this decomposition without the present task-guided investigation.

### Next fidelity test

Replace the native centered cross component t*C(w) with t*C_hat(w), keeping the context-only and squared components intact, and test native output fidelity before any coordinate intervention. Then check whether the same compact operation preserves the causal effects across fixed coordinate doses. This separates a model of an intervention from a faithful representation of a component of the native computation. Use the current fitted parameters without selecting ranks on the replacement results, and retain wrong-family/random controls plus vocabulary-wide error. The copying lexical-coverage learning curve remains outstanding.

### Verification

2,520 full-model forwards. Native and identity logits replay exactly. H890 exact-cross mean effects replay within 0. Discovery support checked against original pair splits; basis orthogonality error 6.66e-15, rank7 discovery reconstruction error 6.58e-15. Exact bilinear recomposition error 1.98e-15; native float-module agreement relative error 8.24e-07.

Independent compiler audit over all120 selected native states: direct context coefficient versus compiled affine encoder max error 3.84e-13, compiled two-direction write versus PCA write relative error 4.67e-16. Compiler and fit source hashes unchanged. A nonfunctional dead branch in the compiler expression was removed and the audit rerun before recording these results. All outputs/fidelity KL finite; no reserved test evaluated; natural-document pool remains26,112. Broad research goal active.
