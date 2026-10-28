## H901 — Discovery-only clustering recovers the useful context partition in the two-template case

The previous turn made progress: H900 froze a within-template covariance rule, showed improvement over four shuffled-label controls on new combinations, but both new grammar templates failed the full-vocabulary competence gate. H901 evaluates those unchanged rules on the earlier panel with a competent grammar subset, and asks whether template labels are necessary for recovering the discovery partition.

### Fixed clustering rule and scope

Use the same32 discovery endpoints from eight regular nouns in templates A/B. Run Euclidean Lloyd K-means with K=2, deterministic farthest-pair initialization, and a100-iteration cap, separately on normalized L12 MLP input activations and exact contextual output writes. No template or number labels are used to fit clusters or select a run. Both rules are retained. Residualize output writes by the resulting cluster means, take their first PC, and decode with the same global mean and exact affine coefficient.

K=2 is a structural prior motivated by the two-template design; it is not inferred automatically. The number axis was already discovered and fixed in earlier task-directed work. Output-write clustering uses that axis and model weights; input clustering does not need the output write to assign discovery clusters, but its subsequent output fitting still uses the fixed axis. This is not end-to-end unsupervised discovery of a computational unit.

Both methods converge without empty clusters: input in2 iterations, output in3. Each yields groups of16/16 and matches template labels perfectly up to permutation; agreement with singular/plural labels is50%. The derived residual direction matches the labeled within-template direction to floating-point precision. Group labels are not used at evaluation time, and no runtime routing is introduced.

### Live validation on the reused panel

Evaluate64 H893 contexts, including competent grammar G and failed-gate grammar H, plus matched copying. Nouns and grammar templates are absent from output discovery, but this panel was repeatedly used in prior diagnostics. The result confirms behavior on a competent subset; it is not untouched confirmation or evidence of generalization to arbitrary datasets.

|Rule|Competent G native KL/removal|Competent G flip-effect RMSE|Pooled grammar flip-effect RMSE|
|---|---:|---:|---:|
|drop_cross|1.000|0.688|1.156|
|pooled1|0.596|1.127|0.897|
|pooled2|0.248|0.273|0.196|
|within1|0.253|0.262|0.189|
|shuffle0|0.596|1.127|0.897|
|shuffle1|0.591|1.120|0.892|
|shuffle2|0.593|1.122|0.894|
|shuffle3|0.608|1.146|0.913|
|cluster_input|0.253|0.262|0.189|
|cluster_output|0.253|0.262|0.189|

On competent G, the frozen within-template rule improves effect RMSE1.127→0.262 versus pooled rank1; all shuffled controls remain1.120–1.146. The cluster-derived rules reproduce this result because they recover the same partition. Native KL/removal falls0.596→0.253. No condition changes G's native top token or binary correctness, reaffirming that answer preservation alone concealed large differences in intervention fidelity.

Pooled rank2 reaches0.273 effect RMSE on G, close to residual rank1, so the grouping rule can recover a useful direction with fewer output coordinates in this case. This is not a matched-total-cost result: grouping is extra discovery computation, and neither setup measures total training or parameter costs of full featurizers.

Grammar G has100% binary preference and56.25% exact-answer top1, passing the prior gate. H has0% exact-answer top1 and fails, retained separately. Copy G/H pass at87.5%/93.75%. Copying effect RMSE improves only0.051→0.047 with the residual rule, and native KL/removal0.737→0.707. This does not establish recovery of copying computation.

### Interpretation and next stress test

A meaningful context partition can be recovered without endpoint labels in this restricted example, and removing between-group covariance exposes a direction with better intervention fidelity. This supports testing conditional/grouped featurizers, rather than interpreting the largest pooled variance direction as the fundamental unit. It still does not validate a manifold: cluster identity could reflect easy prompt/template differences, and the number-related component was supplied upstream.

Next stress the grouping rule on the six-template discovery mixture and multiple fixed K values, retaining all outcomes and measuring whether clusters capture template variation, number, or lexical identity. A grouping that partitions number itself may erase the target signal when residualized. Test this failure mode before generalizing the approach or using clusters as unsupervised task labels. Then extend the same separation of grouping, geometry, and causal tests to other competent computations (language/binding/copying), rather than treating number as representative of the full model. The wider BSF/shared-dictionary/manifold objective remains incomplete.

### Verification

1,472 full-model forwards. Exact replacement is identity with zero logit difference; compiled encoder error1.48e−15. Discovery covariance decomposition and frozen-mean replay remain within1e−15. Source/model/H900 fits unchanged; new cluster fits saved before evaluation and unchanged afterward. Cluster and labeled-rule margins/KL agree within1e−10. All KL finite. No natural or reserved-test evaluation; natural-document pool remains26,112. Broad goal active.
