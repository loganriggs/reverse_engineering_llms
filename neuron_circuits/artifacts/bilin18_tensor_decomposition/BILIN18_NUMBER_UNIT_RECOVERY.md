## H894 — Number decoding recurs across discovery sets, but a unique compact operation is not recovered

Five disjoint noun discovery sets, each with four singular/plural pairs, were fitted separately in two templates. The layer (L12), fitting rule (first input PC, then first contextual-output PC), and evaluation panel were fixed. Labels only orient the input axis and set number centroids; the first PC is never replaced with a better-number PC. All ten fits were evaluated on the same64 synthetic grammar/copying contexts from H893. This is a recovery-stability test on reused evaluation contexts, not fresh confirmation. No natural documents or reserved test were evaluated.

Nine fits classify number perfectly on both their discovery examples and the32 evaluation grammar contexts. The old-transfer nouns in template B instead yield50% accuracy in both. That failure is retained. Low native replacement KL in the failed fit does not validate number recovery: its number-centroid intervention has almost no grammar effect.

### Results by discovery set

The exact flip replaces the fitted number coordinate with the opposite discovery centroid and retains the exact bilinear computation. Effect is the mean reduction in expected-answer logit margin. Approximation RMSE compares exact versus rank1 flip effects, each measured relative to its own native or replaced baseline.

|Fit|Discovery accuracy|Grammar number accuracy|Native replacement KL|Exact flip margin reduction|Rank1 effect RMSE|
|---|---:|---:|---:|---:|---:|
|original_A|1.00|1.00|0.000346|1.143|0.240|
|original_B|1.00|1.00|0.000336|0.605|0.332|
|old_transfer_A|1.00|1.00|0.000386|1.336|0.153|
|old_transfer_B|0.50|0.50|0.000262|0.001|0.068|
|new_regular_1_A|1.00|1.00|0.000515|1.712|0.360|
|new_regular_1_B|1.00|1.00|0.000435|0.962|0.509|
|new_regular_2_A|1.00|1.00|0.000461|1.549|0.376|
|new_regular_2_B|1.00|1.00|0.000492|0.907|0.859|
|irregular_A|1.00|1.00|0.000440|1.488|0.158|
|irregular_B|1.00|1.00|0.000421|0.744|0.833|

Template A is comparatively stable: across its five independent noun sets, oriented number-axis cosine ranges0.849–0.966 (median0.905), and correlations between exact causal-effect profiles range0.888–0.992. However, output-direction cosine is only0.645–0.857, compiled rank1 context-to-output operator cosine0.625–0.826, and relative differences between native approximated cross-term writes are0.346–0.817. The last quantity is asymmetric, normalized by the first fit's stacked write norm; it is not a principal-angle or downstream-fidelity metric.

Template B is more variable, including the retained failed number fit. Across all its pairs, number-axis cosine ranges−0.079–0.944 and compiled-operator cosine−0.001–0.724. Even descriptively excluding that failed fit, grammar causal-effect correlations range0.421–0.979. This exclusion is diagnostic only; it is not a new selection rule or confirmation result. Different centroid distances also mean raw flip-effect magnitudes are not displacement-matched estimates of identical operations.

Copying number accuracy ranges50–87.5% and mean exact margin changes are−0.040 to0.126, versus0.605–1.712 for the nine successful grammar fits. Task labels are used for reporting, not runtime routing. As in H893, grammar template G passes the earlier full-vocabulary competence gate whereas H fails it despite correct binary is/are preference; the pooled grammar panel therefore includes a failed task template. The broad regular/irregular discovery split is helpful, but these small sets are not a representative task census.

### Interpretation

The evidence supports a recurring, context-sensitive number direction more strongly than a uniquely identified rank1 computational unit. A featurizer can classify number perfectly and still recover a materially different contextual write or approximate its causal effects poorly. This is exactly why manifold validation must measure geometry, conditional computation, and interventions separately.

Most informative next diagnostic: inspect whether leading low-dimensional subspaces are stable when individual PCs are unstable, and whether the failed B fit contains number in a lower PC or fails more broadly. Use existing discovery captures for this diagnostic, without changing the ten frozen units or presenting a posthoc better-PC choice as confirmation. Then evaluate all frozen units on the remaining384 locked natural documents, keeping fidelity distinct from weak component influence. Larger balanced discovery panels and displacement-matched interventions remain needed before claiming unique recovery. Broader BSF alternatives, copying coverage, and manifold ontology remain unresolved.

### Verification

2,624 full-model forwards. Exact replacement is an identity with zero logit difference. Maximum dictionary-axis norm error3.55e−15; compiled encoder relative error1.12e−15 against explicit cross-term projections. All recorded KL values finite and source/model hashes unchanged. Fitted units and analyses saved. No reserved test evaluation; natural-document pool remains26,112. Broad research goal active.
