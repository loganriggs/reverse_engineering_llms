## H860 — Reproducible orientation does not establish reproducible feature groups

The prior goal turn made progress: H859 completed a frozen512-document output-fidelity confirmation. This experiment moves from fidelity to the user's manifold-verification requirement. Existing layer4 featurizers show above-random geometric agreement across two seeds, but weak individual group stability. Learned overlapping-group activities are particularly unstable. No architecture earns a semantic-manifold label from these results.

### Protocol and definitions

Use all ten frozen H852 fits, no retraining or task-based selection. A scalar feature is a one-dimensional decoder subspace; blocks and shared groups have eight dimensions. Projector overlap is the sum of squared cosines of principal angles divided by group rank: one means the same subspace, zero means orthogonal. It is invariant to signs and coordinate rotations inside a group. Compute a one-to-one assignment across seeds maximizing total overlap, using dictionaries alone.

Then evaluate the matched groups on all512 H859 natural states. Activity means the magnitude of the decoded vector from that group's active coordinates. Shared groups additionally require their own gate to be active; their overlapping activities do not sum to the reconstruction and are not independent causal attributions. These are native encoder activities, before optional norm calibration.

### Results

|Architecture|Mean matched subspace overlap|Median activity correlation|Median within-token activity correlation|
|---|---:|---:|---:|
|scalar48|0.174|0.191|0.146|
|scalar64|0.124|0.135|0.125|
|block8|0.150|0.262|0.221|
|shared_fixed|0.131|0.103|0.069|
|shared_learned|0.146|0.004|-0.006|

Eight signed ambient-coordinate permutations of the second dictionary preserve all its internal geometry while breaking alignment with the first. Reoptimizing the matching gives mean overlap only0.0085–0.0101 and median activity correlations aroundzero. Thus trained fits share preferred ambient directions, but most matched groups are far from identical. The fraction of groups with overlap greater than0.5 is8.8% for scalar48,5.7% for scalar64,1.6% for block8, and0% for both shared architectures. These thresholds are descriptive, not prevalidated manifold criteria.

A posthoc diagnostic centers each matched activity within token identity. There are197 occurrences in40 repeated-token types;315 singleton tokens contribute zero after centering. Scalar/block correlations remain modestly positive; learned-shared correlation is-.0055. With256 permutations of occurrences within each token type, learned-shared lies inside the central95% null range[-.0280,-.0054], whereas the other architecture medians exceed their corresponding ranges. These are descriptive conditional permutation diagnostics, not uncertainty over training seeds, corrected hypothesis tests, or proof of semantics.

### Relation to known causal directions

Postfit inspection against the discovery-supervised agreement and sentiment axes checks whether the same named behavior is concentrated in a matched component. It usually is not. For block8, the seed1 group with most agreement-axis energy has16.9%, but its geometric match in seed2 has3.8%; for sentiment the corresponding figures are27.0% and1.1%. The axes may be distributed across several differently organized groups.

One exploratory candidate is scalar48's sentiment-associated atom: seed1 atom15 matches seed2 atom126, contains11.4% versus24.0% of the sentiment-axis energy, and has activity correlation0.689. This is not a recovered sentiment feature: selection used a supervised axis, the energy fraction is incomplete, and specificity is untested. Fixed-shared agreement also shows similar axis energies across a matched pair, but only0.170 activity correlation. All candidate results are retained, not just favorable examples.

### What to test next

The leading possibilities are (1) scalar/block dictionaries recover a distributed computational subspace whose partition into groups is nonunique, (2) some genuinely reproducible components coexist with many interchangeable reconstruction directions, and (3) learned shared gates fail to converge to stable partitions, consistent with their earlier membership-coverage collapse. Two seeds and finite training do not distinguish these fully. Above-null orientation can also reflect common covariance rather than reusable computation.

Next use the task-axis-selected scalar48 sentiment pair as an explicitly supervised diagnostic candidate, alongside both tasks' top-axis components and matched random controls. Remove or transfer each component while retaining the reconstruction residual, compare effects across agreement/sentiment and token/distance variants, and keep both seeds. Freeze selection on discovery axes before evaluating effects. Quantify a component's fraction of the full relay effect and off-target changes. If individual components fail but matched unions succeed, test the distributed-subspace hypothesis rather than forcing semantic group names. Binding/copy/capital require their previously verified active sites; do not use absence of early-layer effects as negative evidence about those tasks.

### Verification and limits

SVD-based spans have orthogonality error1.39e-14; self-overlap error6.99e-15. Permuting groups and rotating their internal coordinates retains perfect matches within8.89e-15. Optimal matching is one-to-one and activities do not influence it. All input hashes unchanged, all five architectures included. No full-model forwards or reserved-test evaluations; existing26,112-document pool unchanged. The panel is reused research data, and only two training seeds are available. No causal intervention was performed in H860; that remains the next discriminating step. Broad goal remains active.
