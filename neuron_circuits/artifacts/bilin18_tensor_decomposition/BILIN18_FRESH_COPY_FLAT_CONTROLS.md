## H821 — Fresh flat controls show task-dependent grouping benefits

The frozen shared/private representation is useful, but does not dominate flat subspaces. Complete the missing copying controls using the same H815 frozen uniform basis, 32 fresh lexical items, both source-local states, both response arms, and the previously fixed doses. Reuse H819 full, zero, and shared/private distributions without refitting. Compare these with H820's 31 fresh prefixed indirect-object-identification (IOI) pairs.

### Distributional fidelity and resource costs

KL compares each candidate's final full-vocabulary distribution with the full uncompressed intervention. Divide by the KL incurred by removing the tested L6 module response. Lower is better; these ratios are not fractions of all model computation explained.

| Representation | Stored directions | Active coordinates | Copy KL / zero | IOI KL / zero |
|---|---:|---:|---:|---:|
| Shared + selected private | 8 | 5 | .0947 | .4584 |
| Flat5 | 5 | 5 | .0576 | .5355 |
| Flat8 | 8 | 8 | .0551 | .4258 |

For copying, flat5 improves mean KL from 4.113e-5 to 2.502e-5, and flat8 to 2.392e-5. Paired item-average differences (flat minus shared/private) are -1.611e-5 with descriptive 95% item-bootstrap interval [-3.118e-5,-5.782e-6] and -1.721e-5 [-3.062e-5,-8.393e-6]. Flat5 wins on 20/32 lexical item averages; flat8 on 29/32.

For IOI, flat5 is worse by +2.122e-6 [1.629e-6,2.658e-6]; shared/private wins on all31 item averages. Flat8 is better by -8.962e-7 [-1.459e-6,-4.026e-7], winning on24/31. The new bootstrap uses seed821 and4096 item resamples; small changes from H820 intervals are resampling variation. These intervals are descriptive, not multiplicity-corrected.

Total variation agrees with these orderings. Copying top-1 accuracy is identical (97.656%) across all three representations, illustrating why accuracy alone misses fidelity differences. Zero-module accuracy is slightly higher (98.047%); this is not evidence that deletion faithfully reproduces the original computation.

### What this changes

A single fixed five-dimensional subspace is more efficient and more faithful than the tested shared/private construction on fresh copying. On fresh IOI, adaptive grouping has a fidelity advantage at five active coordinates, but costs eight stored directions; flat8 wins at equal stored directions. Group selection also evaluates the full response against both groups, so active-coordinate counts do not measure total encoder or runtime cost.

These results weaken a universal claim that overlapping groups better match model structure. They support a narrower hypothesis: context-dependent subspaces help some response distributions, while others fit a common low-dimensional subspace sufficiently well. The fixed shared component and alternating private-group fit remain architectural assumptions, not discovered semantic categories. Component ablations establish operational usefulness, not unique or canonical features.

### Implementation verification

2688 new whole-model forwards:128 native local-state recaptures,2048 flat candidates,512 baseline replays. Cached references are reused explicitly. Baseline full logits replay within1.72e-5, below the predefined1e-4 tolerance. Historical full-response scalar replay differs by at most4.65e-6. Dictionary hashes match the pre-native H818 lock and all tracked hashes remain unchanged. Paired source directions match exactly; component-sum error1.44e-15 and basis orthogonality error3.11e-15. KL is finite/nonnegative and full self-KL exactlyzero. No reserved-test evaluations; natural-document pool remains26,112. Full candidate logits and comparison JSON are saved.

### Return to manifold and computation validation

Retain flat and shared/private representations as competing hypotheses rather than declaring an ideal architecture. The present experiment models finite changes entering L6 under one L2H5 perturbation family; it is not yet a native-activation featurizer for L0–L2. It does not establish a hierarchy, computational DAG, or semantic manifold.

Next inspect the existing native-activation featurizer and manifold-validation work before choosing a new experiment. Prioritize a frozen, independent comparison of local geometry and causal transfer: stability across seeds and lexical/surface changes; held-out combinations; component removal and transplantation; and agreement between geometric neighbors and downstream effects. Include flat-subspace and matched random controls. Select task families only after native competence checks. Maintain the diverse-task requirement: copying plus IOI are two narrow probes, not adequate coverage of LLM computation. Do not promote labels or add hierarchy/DAG structure solely because reconstruction improves.
