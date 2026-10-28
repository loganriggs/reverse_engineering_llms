## H899 — Template variance dominates PC1; PC2 restores much of the number-intervention fidelity

The previous turn made progress: H898 demonstrated that the eight-noun A/B rank1 output approximation preserved native answers but distorted number interventions, even beyond removal on competent template G. H899 diagnoses that existing failure without replacing or retraining the fit. The original input number axis/origin and frozen eight_AB output mean/basis remain fixed.

### Exact discovery covariance decomposition

For the32 A/B discovery endpoints, decompose centered contextual writes into template-mean offsets plus within-template residuals. These sum exactly, and their aggregate cross term vanishes. Between-template differences explain56.94% of total discovery variance.

|Output PC|Fraction of total variance|Within this PC: between-template fraction|Within-template fraction|
|---|---:|---:|---:|
|1|0.5813|0.9728|0.0272|
|2|0.1564|0.0252|0.9748|
|3|0.0469|0.0000|1.0000|
|4|0.0418|0.0002|0.9998|
|5|0.0356|0.0003|0.9997|
|6|0.0280|0.0000|1.0000|
|7|0.0215|0.0001|0.9999|

PC1 explains58.13% of total variance and97.28% of its variance is between templates. PC2 explains15.64% of total variance and97.48% of its variance is within templates. This directly supports the hypothesis that pooled variance ordering elevates template differences above a direction needed for the number intervention. It does not establish that template variation is useless to the model or that within-template variance is inherently semantic.

### Live per-PC interventions

For each PC, approximate C(w) with the same global discovery mean plus that PC's exact projected coefficient. Compare mean-only, each individual PC, cumulative first2/7 PCs, exact and cross-term removal. All downstream computation runs live for native replacement and opposite-number coordinate intervention. Individual-PC conditions each include the mean; their behavioral effects are not additive PC attributions.

The64-context evaluation panel is reused from H893/H898 and contains nouns/grammar templates absent from these output fits. The findings are explanatory diagnostics, not fresh confirmation or a selected improved featurizer.

|Approximation|Pooled grammar effect RMSE|Competent G effect RMSE|Competent G native KL/removal|
|---|---:|---:|---:|
|drop_cross|1.156|0.688|1.000|
|mean|0.948|1.190|0.640|
|pc1|0.897|1.127|0.596|
|pc2|0.233|0.321|0.262|
|pc3|0.946|1.189|0.638|
|pc4|0.936|1.161|0.616|
|pc5|0.947|1.174|0.629|
|pc6|0.946|1.189|0.640|
|pc7|0.904|1.140|0.611|
|rank2|0.196|0.273|0.248|
|rank7|0.116|0.159|0.211|

On competent G, PC2+mean reduces effect RMSE1.127→0.321 relative to PC1+mean, while native KL/removal improves0.596→0.262. First2 PCs reach0.273 effect RMSE and first7 reach0.159. The exact mean flip effect is0.843 logits; approximations yield1.736 (PC1),1.121 (PC2),1.086 (rank2),0.985 (rank7). Thus PC2 corrects much of an overestimated intervention effect; it is not proof that PC2 alone implements number.

Mean-only and most individual PCs remain poor. The failure is therefore specifically associated with omitting a useful lower-variance direction, not just too few arbitrary coordinates. Higher PCs collectively still improve fidelity beyond PC2. Downstream sensitivity may also matter: these finite interventions do not separately measure gradients or establish a complete sensitivity decomposition.

Copying remains weakly explained: PC2 lowers native KL/removal0.737→0.710 compared with PC1 and effect RMSE0.051→0.047, versus rank7 0.628/0.042. These small changes are not evidence that the number unit recovers copying computation. G passes the prior grammar competence gate (56.25% exact-answer top1); H fails (0%). Copy G/H pass (87.5%/93.75%). All conditions and failures remain recorded.

### Research direction

The result identifies a concrete extraction failure: broadening a dataset changes which variation PCA prioritizes, and its leading direction can preserve substantial variance while missing a causally useful conditional operation. This connects directly to featurizer design: a group of activations should not be validated solely by explained variance, decodability, or unchanged native answers.

Next test a fixed within-template covariance extraction rule against pooled PCA, with the same mean and matched rank, plus shuffled-template-label controls. Use template labels only as an explicit diagnostic aid; such a rule would not yet supply unsupervised task labels or validate a BSF alternative. Freeze the rule before evaluation on additional lexical/template combinations, and keep native distribution fidelity distinct from intervention fidelity. If the advantage survives, investigate whether unlabeled grouping can recover the same conditional structure. The broader manifold and reusable-computation objective remains incomplete.

### Verification

1,600 full-model forwards. Exact replacement is an identity with zero logit difference. PC1 and rank7 reproduce H898 margins/KL exactly. Covariance energy decomposition relative error1.55e−16; frozen mean replay5.999e−16; compiled encoder projection error1.70e−15. Source/model/frozen-fit hashes unchanged and all KL finite. No natural or reserved-test evaluation; document pool remains26,112. Goal active.
