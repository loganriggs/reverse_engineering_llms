## H776 — Task-informed rank-eight contrasts transfer partially, not uniformly

Previous goal turn was progress: H775 established that small natural-KL gains did not materially preserve copying interventions. H776 asks whether a task-informed representation of the same rank can do better, separating learning-objective limitations from an overly strong claim that rank eight is impossible.

Train on eight predetermined copying pairs: item indices0–3 in semicolon and newline formats. Fit SVD bases to answer-changing activation differences in write geometry, either final-query only or pooled query/source differences. Controls are a basis fitted to nuisance-query differences on those same pairs, ordinary natural write-PCA8, and three fixed random rank-eight write-space subspaces. Every basis has rank eight; the pooled basis is shared across query and source, not eight dimensions per position. No output logits, margin effects or validation outcomes are used to fit or choose a basis.

Test all original32pairs, reciprocal answer changes, nuisance and identity edits at query, source and both positions. Split results by whether item and format were included in fitting. The item holdout is not a vocabulary holdout: cyclic donor pairing causes some answer tokens to appear on both sides of the split. Each crossed cohort contains only four item indices in two formats, with correlated cases.

| Query task cohort | Full effect RMS | Natural PCA8 error | Query-contrast8 error | Pooled-contrast8 error |
|---|---:|---:|---:|---:|
| Training items, training formats | .2350 | .2419 | .0000035 | .0511 |
| Held-out items, training formats | .2043 | .1337 | .1831 | .1813 |
| Training items, held-out formats | .2390 | .2243 | .1283 | .1085 |
| Held-out items, held-out formats | .2669 | .2440 | .2405 | .1799 |

Errors are absolute RMS differences from the full-head edit's answer-margin effect. The query-only training matrix has eight rows, so a rank-eight basis reproduces those training vectors exactly; the near-zero training causal error is an algebraic implementation check, not evidence of discovery.

Held-out transfer is heterogeneous. Pooling query and source contrasts helps on held-out formats: on the doubly held-out cohort its relative effect error is67.4%, versus91.4% for natural PCA and90.1% for query-only contrasts. Random-subspace absolute errors there range.2434–.2566, compared with.1799 for pooled contrasts. But in familiar formats with held-out items, both task-informed bases are worse than ordinary PCA (.181–.183 versus.134). The query-only basis transfers format changes for familiar items substantially better than item changes. None is a reliable universal rank-eight representation.

Vector preservation and effect preservation again diverge. On the doubly held-out cohort, pooled contrasts leave84.9% relative write-vector RMS residual versus80.4% for query-only contrasts, yet preserve the margin effect better. A lower activation reconstruction error alone would select the wrong basis in this comparison. The result is compatible with source contrasts adding behaviorally relevant directions, but it does not by itself establish a reusable source/value computation.

Nuisance controls also expose selectivity limits. On held-out items in familiar formats, the nuisance-trained basis has39.7% relative nuisance-effect error versus71.0% for pooled answer contrasts and76.0% for natural PCA. Conversely, nuisance-trained task error there is.2532, worse than both task-informed and natural bases. This supports distinct contrast-dependent directions on this panel, without proving a clean semantic decomposition or absence of interactions. All nuisance/source/joint outcomes are saved; no favorable subspace is selected post hoc.

Audits:3,072 model forwards; full/PCA margins exactly replay H761; basis orthogonality error<=1.78e-15; training-query relative vector residual1.73e-15; frozen-fit hash unchanged; native identity replay<=2.17e-5; full-donor vector error<=1.77e-5; finite outputs and input/model hashes unchanged. No reserved-test evaluation; natural document pool stays26,112.

Conclusion for the research direction: task-informed directions can improve some held-out causal effects at rank eight, so the earlier failure is not solely explained by rank. But the failure on other item splits and the large remaining errors prevent a verified-manifold claim. This experiment is not a mathematical upper bound on what rank eight can represent and is not an unsupervised featurizer result.

Next: keep these bases frozen and test lexical transfer explicitly. Cross old/new key words with old/new value words across the same formats, using new values disjoint from the eight training value words. Establish native competence before interpreting interventions; include answer-changing and nuisance controls. This distinguishes transferable copying structure from lexical/template-specific directions more directly than the current overlapping-token item split. Report all lexical cohorts, not just the best-performing format.
