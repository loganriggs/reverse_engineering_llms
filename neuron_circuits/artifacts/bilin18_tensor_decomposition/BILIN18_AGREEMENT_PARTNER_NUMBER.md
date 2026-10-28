## H886 — Later agreement partner coordinates track number and matter causally for response fidelity

The rank-1 direction identified by H885 is not uniformly a number direction across layers. At task-context layers L11–16, however, discovery-derived coordinate thresholds classify all eight transfer singular/plural endpoints correctly in both templates. The corresponding copying directions stay at 50%. Replacing coordinates with the correct discovery number-centroid preserves response fidelity much better than using the opposite number-centroid. This supports a context-dependent number-related response component, not a complete agreement circuit.

### Geometry test

Use the existing agreement-only eight noun pairs and the frozen first-four discovery / last-four transfer split. Each layer/family/template receives its own unsupervised rank-1 PCA direction fitted on the eight discovery normalized partner vectors. Fit singular and plural coordinate centroids using discovery labels only. Classify by their midpoint, orienting the sign with the discovery centroid ordering. Evaluate the eight transfer endpoints without choosing layers or thresholds using their labels.

The transfer nouns are report/reports, picture/pictures, bottle/bottles and key/keys; discovery nouns are dog/dogs, cat/cats, book/books and tree/trees. All are regular singular/plural pairs, so grammatical number remains confounded with regular plural morphology. Templates are the previously tested “near the river” and “in the garden” agreement continuations. These are estimation-disjoint nouns, not globally unseen examples, and the two templates were separately fitted rather than cross-template transferred.

|Layer|Agreement A number accuracy|Agreement B number accuracy|Copy A|Copy B|
|---|---:|---:|---:|---:|
|8|0.625|0.500|0.500|0.500|
|9|0.500|0.500|0.500|0.500|
|10|0.500|0.500|0.500|0.500|
|11|1.000|1.000|0.500|0.500|
|12|1.000|1.000|0.500|0.500|
|13|1.000|1.000|0.500|0.500|
|14|1.000|1.000|0.500|0.500|
|15|1.000|1.000|0.500|0.500|
|16|1.000|1.000|0.500|0.500|
|17|0.750|1.000|0.500|0.500|

### Causal coordinate test

Preserve each recipient's native output offset and normalization. Approximate partner vectors by discovery mean plus a rank-1 coordinate at each MLP L8–17. Compare the measured native projection with (a) the correct discovery number-centroid, (b) the opposite number-centroid, and (c) reflection of the measured coordinate about the two discovery centroids' midpoint. All ten layers are intervened together; no posthoc layer subset was selected for this run. Attention remains live and the fixed L8H3 source write is unchanged.

|Condition|Individual native answer-response error|Canonical averaged response error|Vocabulary fidelity KL ratio|
|---|---:|---:|---:|
|linearized|0.013|0.000|0.000|
|own_mean|1.497|0.973|0.350|
|rank1|0.452|0.364|0.269|
|number_correct|0.412|0.426|0.272|
|number_flip|2.514|0.872|1.093|
|rank1_flip|2.237|0.926|0.719|

The correct number-centroid gives 41.2% individual error, comparable to native rank-1 projection's 45.2% and far below the mean's 149.7%. Opposite-number centroids give 251.4%; coordinate reflection gives 223.7%. Vocabulary fidelity likewise worsens: the correct-centroid KL ratio is 0.272 versus 1.093 for opposite-number centroids. This ratio compares KL(live edited || approximated edited) with KL(live edited || unedited native), so the wrong-number approximation is worse than the no-response reference under that metric.

Canonical averaging again understates some errors: wrong-number centroids have 87.2% canonical error, below the mean's 97.3%, despite being much worse on individual interventions. The conclusion relies on signed individual responses and full-distribution fidelity, not merely average margin matching.

### What is established and what is not

A discovery-derived low-dimensional partner coordinate in later agreement contexts carries a transferable number association, and number-conditioned coordinate values influence the fidelity of the measured source-write response. This is stronger than labeling a principal component from correlation alone. But the operation changes a local derivative around cached native offsets, not native agreement behavior by itself: identity interventions preserve native outputs. It does not show that erasing this coordinate makes the unperturbed model lose agreement, or that flipping it reliably flips a verb prediction.

The fixed reader's agreement response remains weak, and correct-centroid approximation retains roughly 41% individual error. The coordinate is therefore a candidate component of conditional computation, not a complete unit ontology. All ten layers were manipulated, including layers whose rank-1 coordinate did not generalize number; the present experiment does not uniquely localize causal necessity to L11–16. Chance copying classification concerns only copying's first PC, not the absence of number information in its other dimensions.

### Next discriminating experiments

Freeze the discovered directions and number centroids and test irregular singular/plural nouns, plural-looking singular nouns where suitable, and fresh agreement templates with number-mismatched distractors. Screen full-vocabulary competence before causal interpretation. This distinguishes grammatical number from an s-suffix or template-correlated direction. Separately compare intervention on the prespecified L11–16 set with its complement to determine whether the geometrically identified coordinates account for the causal effect. Preserve individual native-fidelity and vocabulary-wide measures; do not promote a coordinate based only on label separability. The copying learning-curve question from H885 remains open and is not resolved by this agreement-focused experiment.

### Verification and numerical qualification

Final audited run: 1,408 full-model forwards. Identity final states are exact. All 40 fit supports are discovery-only. Jacobian versus autodiff relative error 3.36e-16; finite-difference error 1.08e-08; all ten layers checked. Basis orthogonality error 1.15e-14. Selected readout error 7.29e-06.

The initial run rejected the absolute cross-run final-state replay tolerance: maximum coordinate difference was 0.0585938. The agreement-only subset changes batch shapes. The repeated audited run retained this absolute discrepancy and added relative-state and selected-logit checks: relative state error 9.14e-07, selected logits difference 5.67e-06. These support numerical agreement of the measured effects rather than claiming bit-exact cross-batch states. Identity fidelity KL is zero; outputs and KL values finite; source hashes unchanged. No amplitudes dropped or reserved tests evaluated. Natural-document pool unchanged at26,112; broad goal active.
