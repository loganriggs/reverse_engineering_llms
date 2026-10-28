## H777 — Frozen contrast subspaces do not preserve copying across new vocabularies

Previous goal turn was progress: H776 found partial format/item transfer but identified overlapping answer tokens as a limitation. H777 explicitly crosses lexical novelty while keeping every subspace frozen.

The panel crosses old color keys versus new animal keys with old tool/object values versus new fruit values, in four formats: semicolon, newline, colon and equals. Eight item pairs per cell produce128pairs/384native contexts. Answer-changing edits retain the key and replace the value; nuisance edits rename the key while retaining the intended answer. The new value continuation-token IDs are disjoint from all eight old value tokens. No tokenization failures occurred and no examples were excluded. Old/old is a replay control containing fitting examples, not novel evidence.

All16lexical-cohort/format cells pass the prespecified native competence gate. Paired intended-answer choice and nuisance correctness are100%; full-vocabulary intended top1 is87.5–100%. Thus the tested model generally performs the new copying tasks, and weak featurizer transfer cannot be attributed simply to native failure.

The H776 query-contrast, pooled query/source, nuisance-query, natural-PCA and three random rank-eight subspaces remain unchanged. We evaluate query-only, source-only and joint F(donor)-F(receiver) edits against full-head references. The following aggregates all four formats for query task edits; absolute RMS errors are answer-margin errors.

| Keys / values | Full effect RMS | PCA error | Query-contrast error | Pooled-contrast error |
|---|---:|---:|---:|---:|
| Old / old | .2373 | .2158 | .1642 | .1411 |
| New / old | .2737 | .2327 | .2323 | .2267 |
| Old / new | .3183 | .2912 | .2997 | .2811 |
| New / new | .3023 | .3210 | .2787 | .2890 |

For the three novel lexical cohorts, pooled-contrast relative effect errors are82.8%,88.3% and95.6%, respectively. The query-only basis likewise has84.9%,94.2% and92.2% errors. These are not faithful copying representations despite occasional improvement over PCA. Old/old has a full effect below the existing.25relative-error threshold, so its relative ratio is not reported; moreover, its aggregate contains the fitting pairs and should not be compared as an unbiased generalization estimate.

Write-vector residuals also grow: pooled task differences retain relative RMS residuals81.3% for new keys/old values,88.6% for old keys/new values and92.3% for both new. Nuisance effects remain non-negligible and imperfectly preserved. The nuisance-fitted subspace does not establish a clean, vocabulary-independent decomposition either. All formats, positions, controls and vector residuals are retained rather than selecting favorable cells.

The lexical intervention changes semantic category as well as word identity (colors to animals, tools to fruits). Consequently this is evidence against generality of these particular frozen bases, not a clean estimate of a universal lexical-novelty effect. More vocabulary sets could separate category effects, but the current basis already fails the practical transfer criterion. New words here means disjoint from the fitting vocabulary, not absent from pretraining.

Audits:12,672 model forwards; all128planned pairs valid; new-value token disjointness verified; frozen-fit and model/input hashes unchanged; basis orthogonality<=1.78e-15; old/old margin replay<=1.15e-5; identity logits replay exactly; full-donor vector error<=9.54e-7; all outputs finite. Original reserved test untouched; natural document pool remains26,112. The new synthetic prompts do not increase that natural-document count.

Research interpretation: the H776 contrast basis captured some lexical/template-specific structure but is not a verified reusable copying manifold. This does not prove that copying itself requires a complex computation. A compact routing rule could act on high-dimensional word-identity content, making a global low-rank head-output representation the wrong object to seek.

Next information-gaining test: inspect the actual L5H5 attention implementation and separate routing changes from carried-value changes in matched copying edits. For aligned contexts, decompose donor-minus-receiver head output into routing-only, value-only and their interaction, and audit that these sum to the full head difference before any causal interpretation. Then intervene on the components with downstream computation live across old/new lexical cohorts. This can distinguish a reusable routing operation with high-dimensional content from a genuinely inseparable task-specific transformation. Nonlinear downstream effects must be reported directly; the component margin effects need not sum and are not unique mediation fractions.
