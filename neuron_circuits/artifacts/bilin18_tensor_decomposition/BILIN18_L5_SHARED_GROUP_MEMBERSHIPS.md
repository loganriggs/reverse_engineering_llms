## H757 — Shared groups contain structure, but do not yet improve the featurizer

This test extends each of the six frozen H755 scalar dictionaries with128 overlapping groups of four atoms. Each group retains its own anchor atom; total memberships are fixed at512. Two distinct groups are selected and decoded through their exact union, so shared atoms contribute once. Fixed counts constrain membership capacity instead of fitting an L0 membership penalty. Decoder columns remain normalized through the frozen scalar model.

Memberships are initialized from either stable-half training support correlation or random anchored groups, then updated for ten rounds to maximize captured raw coefficient-squared energy. Assignment uses a greedy start and alternating best responses; membership updates exactly optimize each anchored row for fixed assignments. The best training-scored snapshot, including initialization, is retained. This objective is a proxy for reconstruction because decoder atoms are nonorthogonal. It is not joint dictionary/gate training or a global optimum.

| Head | Scalar8 | Union-size-matched scalar | Initial correlation groups | Learned correlation groups | Degree-preserving randomizations |
|---|---:|---:|---:|---:|---:|
| L5H3 | 41.16% | 41.22% | 52.71% | 52.68% | 56.09% |
| L5H5 | 38.91% | 38.98% | 50.21% | 49.66% | 52.65% |

Values are natural validation relative write MSE, averaged over all three dictionaries; randomization values average three rewires per dictionary. Learned random-initialized groups reach55.53%/51.48% at H3/H5, versus their randomized controls56.55%/52.87%. All initializations and nulls remain in the results. Validation is evaluated, not used to select memberships.

The group structure improves over degree-preserving nulls, especially with correlation initialization, but remains substantially worse than the frozen scalar Top8 featurizer. A per-example scalar control selects exactly as many atoms as the group's union; its error remains almost unchanged. Thus the gap is not explained by losing active slots to overlap. Group constraints and/or the energy-proxy optimizer remain plausible bottlenecks.

Learned correlation groups select7.975(H3)/7.982(H5) unique atoms on average. The two selected groups share atoms in only2.47%/1.77% of validation examples. This does not rule out reuse of dictionary atoms across different contexts, but simultaneous sharing is rarely exercised. Overlap as an architectural capability is not evidence of shared computation or a meaningful manifold.

Degree-preserving rewires maintain every group's size and each atom's membership count, but do not preserve the anchor designation. Anchors are a training constraint, not a special decoder role. The comparison controls membership degrees and capacity; it does not eliminate every possible structural confound.

Implementation checks: exact union agrees with binary matrix-product support over every validation example; two gates are selected and at most eight unique atoms decoded; rewiring preserves atom and group degrees; fixed-assignment membership energy never decreases. Seven targeted tests pass, including a known shared-atom example, exhaustive small membership optimization, anchored row sizes, assignment monotonicity and atom-permutation invariance. The actual H757 membership functions were used in the tests and matched the previously audited function sources. All group sizes remain four; inputs/source hashes unchanged and outputs finite.

No model forwards or original reserved-test evaluations occur here; natural document pool remains26,112. These are frozen-dictionary overlapping supports, not evidence against all jointly trained shared-gate, hierarchical or computational-DAG alternatives.

Next: evaluate learned correlation groups and all degree-preserving nulls on the existing total-effect and operation-dependence causal targets, with all scalar seeds retained. Only a reproducible functional advantage would justify treating these groups as better candidate units; otherwise diagnose gate/objective restrictions before adding hierarchy.
