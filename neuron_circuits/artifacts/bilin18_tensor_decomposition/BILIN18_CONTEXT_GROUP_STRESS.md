## H902 — Unlabeled grouping can remove the signal it should explain

The previous turn made progress: H901 recovered the two-template discovery partition without labels and reproduced a useful residual direction on a competent grammar subset. H902 stresses that result with96 endpoints from eight regular noun pairs across six discovery templates. The input number axis remains fixed. All K=2,4,6,12 clusterings are retained in normalized input space and contextual output-write space.

Use deterministic farthest-pair initialization, then farthest-point additions for K>2, Euclidean Lloyd updates, and a100-iteration cap. Labels do not determine clusters or choose K. All eight clusterings converge without empty groups. After grouping, residualize output writes by group means, fit rank1, and use the same global mean at runtime. References are pooled ranks1/2/7 and rank1 after labeled template residualization. All fits are frozen before live evaluation on the reused64-context panel; no evaluation-based best-K selection.

### Discovery groups and retained contrast

Label entropy explained is I(cluster;label)/H(label), measured after fitting:0 means no empirical association,1 means the cluster identifies the label. These are descriptive in-sample statistics, not chance-adjusted clustering scores. The contrast ratio is the norm of the singular/plural mean-output difference after residualization divided by that before. It is a centroid diagnostic, not total number information, and does not imply that all number information has disappeared.

|Grouping|Template entropy explained|Number entropy explained|Noun entropy explained|Number centroid contrast norm retained|
|---|---:|---:|---:|---:|
|cluster_input_2|0.251|0.000|0.000|1.000|
|cluster_input_4|0.742|0.000|0.000|1.000|
|cluster_input_6|1.000|0.000|0.000|1.000|
|cluster_input_12|1.000|0.833|0.015|0.121|
|cluster_output_2|0.139|0.189|0.010|0.482|
|cluster_output_4|0.464|0.169|0.002|0.598|
|cluster_output_6|0.822|0.189|0.003|0.575|
|cluster_output_12|1.000|0.456|0.076|0.220|

Input K=2/4 group whole templates or collections of templates; K=6 exactly recovers all six templates, with no empirical number/noun association. Input K=12 identifies templates but additionally explains83.3% of number-label entropy, leaving only12.1% of the original number-centroid contrast norm in the residuals. Output clustering mixes template and number earlier: K=2 already explains18.9% of number entropy and leaves48.2% of the contrast norm.

This invalidates a blanket rule that cluster means are nuisance variation to remove. A data-dependent partition can put a target computation between groups. Its geometry can still look plausible: input K=12's residual leading direction has cosine0.954 with the labeled within-template direction, despite worse downstream behavior. Direction cosine alone misses that distinction.

### Live fidelity on the competent grammar subset

Native KL ratio is relative to removing the same exact cross term. Effect RMSE measures exact versus approximated number-flip margin changes, each relative to its own baseline. G is the previously competent template; H remains a failed-gate template and is retained separately in the full analysis.

|Rule|G native KL/removal|G flip-effect RMSE|Pooled grammar flip-effect RMSE|
|---|---:|---:|---:|
|drop_cross|1.000|0.688|1.156|
|pooled1|0.245|0.258|0.194|
|pooled2|0.231|0.229|0.170|
|pooled7|0.199|0.151|0.108|
|within1|0.231|0.178|0.129|
|cluster_input_2|0.254|0.255|0.193|
|cluster_input_4|0.239|0.189|0.137|
|cluster_input_6|0.231|0.178|0.129|
|cluster_input_12|0.260|0.312|0.235|
|cluster_output_2|0.548|1.066|0.847|
|cluster_output_4|0.292|0.440|0.326|
|cluster_output_6|0.248|0.288|0.216|
|cluster_output_12|0.261|0.330|0.244|

Input K=6 reproduces labeled residualization and improves G effect RMSE0.258→0.178 relative to pooled rank1; K=12 worsens it to0.312. Output K=2 gives1.066, worse than removal0.688, despite native KL still below removal. No K value is selected as a deployment rule. Knowing that the discovery design contains six templates cannot be presented as an unsupervised estimate of six meaningful groups.

Pooled rank7 remains stronger at0.151 G effect RMSE. Copying effect RMSE stays around0.039–0.042 across most conditions; clustering has not revealed its mechanism. Grammar G passes the earlier gate at56.25% exact-answer top1, H fails at0%; copy G/H pass at87.5%/93.75%. These reused, small panels remain diagnostic rather than independent broad validation.

### Consequence for featurizer design

The next design should preserve between-group structure as well as within-group variation, rather than assume group means can be discarded. This connects to the user's shared-dictionary/hierarchical proposal: groups can select or refine reusable components without deleting computations carried by their differences.

Next test a fixed input-cluster conditional approximation that retains cluster-mean contributions alongside shared residual directions, including the failed K=12 case. Compare with global-mean residualization, cluster-mean-only, and appropriate capacity baselines; count additional means/gating parameters explicitly. Runtime cluster assignment must use only activations and frozen discovery centroids, with all K values retained. Test transfer and causal fidelity before interpreting clusters as task labels. Other competent computations must still be included before drawing a model-wide conclusion; number is a case study, not a complete ontology.

### Verification

1,856 full-model forwards as planned. All eight clusterings converge. Exact native replacement has zero logit difference; compiled projection error1.49e−15. Covariance decomposition error9.29e−17; frozen mean replay3.64e−16. Source/model/fits unchanged and new fits frozen across evaluation. All KL finite. No natural or reserved-test evaluation; document pool remains26,112. Goal active and broader feature/manifold objective incomplete.
