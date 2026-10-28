## H584 — One discovery direction preserves most of the agreement swap despite poor internal reconstruction

Previous goal turn made progress by replicating a strong Attention5 agreement effect. Fit an uncentered SVD to eight discovery plural-minus-singular Attention5 write differences at the final “usually” token. Freeze/save the basis before interventions. Evaluate ranks 1, 2, 4 and 8 on every discovery and lexical-transfer endpoint. Also test the mean-direction projection, a fixed signed mean write, a basis from discovery same-answer nuisance differences, and five seeded random bases. Nuisance/random projections are rescaled to the PCA perturbation norm at each rank/case, so their weakness cannot be attributed simply to smaller perturbations. Full swap and zero controls retained.

2,016 interventions cover counterfactual and nuisance donors. Projected coefficients use the actual donor difference: this is a compact representation of a known intervention, not a learned mechanism that generates the coefficient from the prompt. The fixed mean is a stronger coefficient-free test and is only applied to counterfactual swaps.

Use the live full swap as reference: KL(full-swap distribution || approximate distribution); absolute margin error divided by mean absolute full-swap effect; intended-answer-group NLL error. Report internal norm error separately. The no-intervention distribution is the baseline for relative KL. Predeclared descriptive fidelity criterion requires both relative margin error and relative reference KL <=10%; passing does not establish semantics or a canonical basis.

| Transfer approximation | Internal relative error | Relative margin error | Relative reference KL | Absolute reference KL | Donor-directed margin |
|---|---:|---:|---:|---:|---:|
|full|0.0000|0.0000|0.0000|0.0000|2.1581|
|zero|1.0000|1.0000|1.0000|0.2140|0.0000|
|fixed_mean|0.5054|0.1593|0.1084|0.0232|2.4365|
|mean_projection|0.4778|0.0772|0.0782|0.0167|2.1104|
|pca_1|0.4802|0.0821|0.0796|0.0170|2.0870|
|pca_2|0.4724|0.0745|0.0683|0.0146|2.1034|
|pca_4|0.4389|0.0670|0.0488|0.0105|2.1543|
|pca_8|0.4041|0.0568|0.0436|0.0093|2.1791|
|nuisance_matched_1|1.2927|0.9476|0.9878|0.2114|0.1130|

Across five norm-matched random rank-one controls, relative margin error ranges 0.947–1.022, and relative reference KL ranges 0.941–1.049. All seeds remain in the result.

Rank one captures 86.2% of discovery difference energy, but held-out internal relative error is still 0.480. Nevertheless, its relative margin error is 0.082 and relative reference KL 0.080. Mean-direction projection also meets the descriptive criterion. A fixed mean write misses it (margin error 0.159, relative KL 0.108): context-dependent magnitude matters. Higher ranks improve agreement fidelity modestly while still leaving substantial held-out internal error.

This is a concrete example of the user's fidelity spectrum: a representation can be poor at reproducing the whole internal write while preserving much of a particular causal effect. It does not make the omitted directions globally irrelevant. On nuisance swaps, rank-one PCA leaves relative reference KL about 1.002 and relative margin error 1.058: it fails to reproduce those smaller effects. Their absolute reference KL is about 0.00216, compared with 0.01703 for rank-one counterfactual approximation. Whether that loss is acceptable depends on the fidelity objective; relative and absolute metrics tell different stories.

The evidence supports a number-associated causal direction in this operation/wording, not a universal number feature. Only eight noun pairs per split and their paired directions are tested; all use “usually”. Direction choice was discovery-only, but the operation/site was selected in prior experiments. No independent population confidence claim is made.

Next H585 freezes the same basis and tests other agreement wordings, including a different final token, and the other diverse task contrasts without refitting. Carry over competence gates and failed cells. Cross-context projection fidelity, not compactness alone, determines whether the proposed unit is reusable.

Audits: full swap replays H583 within 1.72e-5 margin; zero <=1.24e-5; basis orthogonality error <=1.08e-6; norm-matched retained-norm ratios agree within 1e-6 (raw absolute norm rounding <=1.23e-4); source/basis unchanged; finite metrics. No new prompts or natural-text documents. FineWeb total 26,112. Broad goal active and incomplete.
