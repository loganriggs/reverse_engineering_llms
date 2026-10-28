## H671 — Suppressive alignment is present before later branches

We compared the full finite response to increasing layer14 head4's projected input value with its direct residual path to the final readout. The same128 H670 contexts span nouns/plurals, adjectives/antonyms, countries/capitals and months/succession, with copying and task instructions, four held-out items and four forms each. This is a retrospective analysis of existing contexts, not fresh-data replication.

The direct direction multiplies the projected value by the product of residual coefficients in blocks15–17 (1.443206787109375), then differentiates the final RMS normalization, unembedding and tanh saturation at the native final state. It omits changes in MLP14 and subsequent attention/MLP branches. Full effects use the saved central finite intervention at ±0.05; ±0.025 checks step sensitivity. Raw unembedding alignment is also inspected, without interpreting its scale as an output effect.

All128 contexts have negative current-input logit alignment in the raw projection, negative direct readout slopes, and negative full slopes. Thus later branches are not required to create the observed suppressive sign in these local diagnostics.

| Family | Direct input slope | Full input slope | Direct answer slope | Full answer slope | Vocabulary cosine | Direct/full norm | Relative error |
|---|---:|---:|---:|---:|---:|---:|---:|
|plural|-4.1388|-2.7083|-3.9532|-2.6048|0.9676|1.4926|0.5830|
|antonym|-2.1281|-1.7349|-0.3188|-0.0871|0.9518|1.1685|0.3979|
|capital|-3.9041|-3.2039|-2.3009|-1.7846|0.9796|1.2083|0.3031|
|successor|-2.5851|-2.4410|0.2263|0.6249|0.9827|1.0145|0.1910|

Vocabulary comparisons center each response by its native-probability-weighted mean and use the native probability distribution as the weighting. Cosine measures directional alignment; relative error is norm(full−direct)/norm(full). These are weighted local metrics, not an unweighted guarantee for rare tokens. Table entries are means over32 contexts per family, pooling copying and task instructions.

Later branches attenuate mean input suppression in all four families, most strongly for nouns. High cosine does not imply numerical equivalence: mean relative error is0.583 for plurals and0.191 for months. Related plural and capital answers are already suppressed by the direct path. For months the full next-month answer slope is0.625 versus0.226 directly, indicating that later processing materially changes the alternative-token response even when the overall direction aligns closely.

The leading explanation is a context-dependent gate applied to an input-associated value direction that already aligns with suppression at the output. This is stronger evidence against interpreting this vector as a dedicated successor operator. It does not establish that suppression is beneficial, that the actual signed coefficient always suppresses, or that this head is necessary. Increasing the coefficient is the tested intervention; a negative coefficient can reverse the effect.

The direct path is an analytic diagnostic, not a finite branch ablation. Full−direct approximates the combined downstream derivative near zero and must not be read as causal shares. The value itself contains local and persistent-bus contributions. The next informative experiment is to separate those two value sources before projection, test whether their sum reconstructs the measured direction, and compare their direct and full responses across the same diverse domains. A fresh-item replication and stronger random-direction controls remain needed before claiming broad generality.

Audits: no new whole-model forwards. Native readout replay maximum absolute error1.609e-5; analytic direct derivative versus finite final-readout difference maximum relative error7.332e-6; full ±0.05 versus ±0.025 step discrepancy maximum relative error2.612e-4. All input and runner hashes unchanged, finite results. FineWeb remains26,112 unique documents. Broad goal active and incomplete.
