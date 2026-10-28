## H865 — Diverse-task contrast erasure distinguishes value-bus dependence from residual/route interactions

H864 found that a single layer4 relay coordinate contributes to agreement/sentiment but is not required for most correct answers. H865 broadens the necessity diagnostic to all five competent task families at the previously verified early boundary. Binding, capital retrieval and copying strongly depend on value-bus contrasts under this intervention; agreement and sentiment show more distributed route dependence and interaction.

### Design and metric

Use all76 aligned H843 task pairs and both recipient contexts at the boundary after block1. Keep three persistent state components separate: residual sequence R, layer0 value bus V, and normalized embedding anchor E. For each of all8 subsets, replace the selected components with the elementwise pair midpoint in both contexts; retain native unselected components. Run blocks2–17 and the output head. No featurizer or supervised direction is fit. This is a controlled contrast-erasure factorial on a reused panel, not a new discovery of the persistent value route already studied in H844 and earlier work.

Primary outcome is the fraction of the native paired answer-margin distinction removed. Both examples use identical answer groups. This avoids an easy denominator error: the runner's inherited per-recipient gap_fraction summary is half the paired contrast removal. Use the explicitly paired analysis below. It is not a unique natural mediation fraction; erasure can create off-manifold states, and route interactions make additive causal attribution inappropriate.

### Results

|Task|R erased|V erased|E erased|R+V erased|Binary accuracy, native → V erased|
|---|---:|---:|---:|---:|---:|
|agreement|32.7%|4.3%|6.2%|65.8%|100.0% → 100.0%|
|capital_recall|0.5%|90.5%|1.8%|92.7%|96.2% → 76.9%|
|entity_binding|1.2%|97.0%|-0.1%|99.4%|100.0% → 59.4%|
|sentiment|28.6%|25.1%|4.3%|83.3%|100.0% → 100.0%|
|sequence_copy|9.3%|84.9%|3.4%|93.1%|100.0% → 68.8%|

Equalizing V removes97.0% of binding,90.5% of capital-retrieval and84.9% of copying distinctions. Corresponding binary accuracies fall to59.4%,76.9%,68.8%. Native capital accuracy is96.2%, not100%; the existing task family passed its competence gate without every pair being correct. Margin loss and accuracy need not track proportionally because residual margins and biases can preserve the answer after a large reduction.

Agreement behaves differently: R erasure removes32.7% of the native distinction, V4.3%, but jointR+V65.8%. Sentiment loses28.6% underR,25.1% underV, and83.3% jointly. These combined effects exceed the sum of separate losses and support route interaction or compensating information under the chosen intervention. They do not establish a unique redundancy decomposition. E also interacts: agreementR+E loses60.4% despite separate losses32.7% and6.2%.

Equalizing allR,V,E makes paired suffix logits exactly identical, removes100% of every task's paired distinction, and yields50% binary accuracy. This is a completeness check on the captured suffix state, not an independent discovery about model representation. Native replay and identical-answer-group checks pass.

### Implication for the manifold program

A useful representation must be evaluated where a computation actually depends on it. Residual featurizers at the layer4 relay remain useful diagnostics for agreement/sentiment, but their successes or failures cannot settle feature organization for binding/copying/capital retrieval. The bus provides a stronger causal target for those tasks.

The layer0 value bus is token-local storage. Strong dependence on it does not mean it contains an already-contextualized binding relation. Downstream attention readers combine stored values with context-dependent routing. Any manifold/feature interpretation must distinguish lexical content directions from reader selection and transformation. Prior reader/bus studies already establish parts of this mechanism; reconcile new tests with them instead of claiming rediscovery.

Next localize contrast dependence within the9 value-head slices, keeping allheads and allfive tasks, before committing to another large featurizer fit. Compare single-slice midpoint erasure with complete-bus erasure, preserve native residual/anchor, and report nonadditivity and per-task correctness. Then relate shared or task-specific slices to existing reader interventions and test held-out lexical/query combinations. A token-identity code or a head partition alone is not a semantic manifold or reusable computational DAG.

### Verification and limits

1,216 suffix forwards; no new whole-model forwards. Native replay logit error4.48e-05; paired all-state-midpoint logits match exactly. All76pairs retained; paired answer groups agree. Source hashes unchanged, metrics finite, original reservedtest untouched,26,112-document pool unchanged. Discovery/transfer split summaries and all8 masks are saved.4096 pair-bootstrap intervals are descriptive and uncorrected; shared templates and lexical dependencies can make them optimistic. They are not evidence of independent template-population generalization. Broad research goal remains active.
