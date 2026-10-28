## H905 — Paired contrasts help binding geometry, but L12 has weak language/binding margin influence

The previous turn made progress: H904 showed that stable leading PCs can miss held-out task-distinguishing changes in language and binding. H905 fits both pooled centered activation PCs and uncentered SVD subspaces of four discovery paired-change vectors, at ranks1/2/4. The latter uses pair-construction supervision but does not treat endpoint order as a universal semantic label: reversing a difference vector leaves its contribution to the SVD subspace unchanged. A seeded random rank4 control is retained.

Each domain/template uses four task discovery pairs and four disjoint transfer pairs. Test both task and matched copying,96 endpoints total, using matching-template task-trained bases. These are reused diagnostic data.

### Finite intervention definition

At final-query L12, form the difference from the recipient's live normalized MLP input to its matched opposite-source donor's cached normalized input. Apply the full difference, its subspace projection, or zero, through the exact bilinear finite expansion including its quadratic term. Only the MLP output is changed; the recipient residual, attention, and downstream context remain otherwise live. The full intervention is an MLP-only donor substitution, not a full prompt/state swap or a necessity test for all task computation.

Projected normalized inputs are not renormalized. Task paired-subspace interventions change their norm by at most about0.2% in the reported rank1/4 conditions; this is a limitation, not a norm-matched causal isolation claim. Full changes preserve the donor normalized norm to numerical precision.

### Held-out task geometry and intervention fidelity

Capture is mean fraction of paired-change squared norm retained. Effect RMSE compares each approximation's expected-answer margin change against the full MLP donor substitution. The no-change RMSE measures the full substitution's RMS margin effect and is essential for interpreting small errors.

|Domain|Fit|Change energy captured|Effect RMSE vs full substitution|No-change RMSE|
|---|---|---:|---:|---:|
|agreement|pooled_1|0.586|1.127|2.166|
|agreement|pooled_4|0.678|0.710|2.166|
|agreement|paired_1|0.626|0.927|2.166|
|agreement|paired_4|0.678|0.754|2.166|
|language|pooled_1|0.041|0.089|0.120|
|language|pooled_4|0.137|0.076|0.120|
|language|paired_1|0.037|0.090|0.120|
|language|paired_4|0.128|0.072|0.120|
|binding|pooled_1|0.023|0.073|0.070|
|binding|pooled_4|0.491|0.059|0.070|
|binding|paired_1|0.351|0.059|0.070|
|binding|paired_4|0.490|0.051|0.070|

Binding rank1 capture improves2.3%→35.1% with paired discovery; its rank4 capture is similar to pooled rank4 (~49%). Agreement improves modestly at rank1 but not consistently at rank4. Language remains low even at rank4 (~13%), so four paired training differences do not reveal a broadly shared low-dimensional source-change subspace there.

Critically, the full L12 substitution's RMS task-margin effect is2.166 for agreement, but only0.120 for language and0.070 for binding. Mean signed effects are1.903,−0.007 and0.010 respectively; the latter two means can hide cancellation, hence the RMS reporting. Full substitutions still change complete output distributions (mean native-to-edited KL0.02668,0.00225,0.00125 respectively), so weak target-margin effect is not zero global influence.

Native exact-answer top1 on the task transfer endpoints is87.5% agreement,100% language,87.5% binding. Thus weak language/binding margin effects are not explained by a wholly incompetent task panel. They may reflect layer/site choice, redundancy, downstream compensation or the restricted MLP-only intervention. This experiment does not distinguish those explanations.

### Copying and generalization limits

Task-trained subspaces mostly fail to capture matched copying changes: rank4 energy capture remains below3.1%. Their approximation effect errors are close to doing nothing. Full copying substitutions have RMS margin effects0.732 agreement-copy,0.277 language-copy and0.216 binding-copy, demonstrating nonzero behavior left unexplained by these task fits. The task-derived directions do not establish shared copying computation.

The geometry gain in binding is useful, but cannot alone validate a fundamental computational unit when the corresponding full L12 intervention has weak target influence. Nor does poor language rank4 transfer imply the model lacks structured language computation; its relevant variation may be larger, differently localized, or absent from this small discovery set.

### Next evidence

Localize before further feature fitting. Use the already cached L8–17 states to screen full matched-source MLP substitutions, with all layers reported, task/copy controls, target-margin RMS and full-distribution KL. Separate a single-layer site from coordinated multi-layer interventions. This will identify whether L12 inherited from the number study is simply an uninformative probe for language/binding, and whether upstream/context pathways need attention instead. Any subsequent layer selection requires new or explicitly reused validation; do not relabel the screening data as confirmation. Broader manifold and reusable-feature work remains active.

### Verification

960 full-model forwards. Zero-change intervention is identity to full logit precision. Cached recipient activations replay with relative error4.70e−7; bases are orthonormal to7.77e−15. An independent direct bilinear evaluation versus finite-expansion audit on96 cached endpoint pairs agrees to relative4.70e−15. All source/model/frozen-fit hashes unchanged and all recorded KL finite. No natural or reserved-test evaluation; document pool remains26,112. Goal active.
