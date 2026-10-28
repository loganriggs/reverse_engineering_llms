## H839 — Some scalar directions recur, but learned shared gates do not form stable labels

Use frozen H836 models and H837's512 new-document states to compare seeds8351 and8352. Match scalar directions, block8 subspaces and shared8-atom group subspaces with Hungarian one-to-one assignment maximizing mean squared principal cosines. Compare binary activation supports under those geometric matches. This is exploratory reuse of confirmation activations, not a new independent confirmation or semantic-label test.

### Cross-seed stability

All-unit means; overlap is squared principal-cosine overlap (rank1 reduces to squared direction cosine). Jaccard is support intersection/union.

| Site | Model | Geometric overlap | Support Jaccard |
|---|---|---:|---:|
| block1 | scalar48 |.221|.159|
| block1 | scalar64 |.154|.147|
| block1 | block8 |.152|.181|
| block1 | fixed shared |.162|.061|
| block1 | learned shared |.163|.036|
| block2 | scalar48 |.196|.152|
| block2 | scalar64 |.129|.137|
| block2 | block8 |.158|.198|
| block2 | fixed shared |.148|.062|
| block2 | learned shared |.174|.033|

Five signed ambient-coordinate permutations of the second decoder, each with a separately optimized assignment, give mean geometric overlap around.008–.010. Scalar support Jaccard under these reference assignments is.038–.058, block8 .042–.046, fixed shared .030–.032 and learned shared .031–.032. These references preserve each model's internal geometry and original activation masks; they are not five independently trained models or formal significance tests.

Decoder geometry agrees beyond these references for every family. Learned shared gate support, however, is barely above reference agreement despite decoder-subspace alignment. Shared group labels therefore should not be treated as stable computational categories. Fixed random group graphs differ across seeds and create a different identifiability problem; their partial agreement is not learned semantic group recovery.

Scalar48 directions are more reproducible than scalar64, despite scalar64's better local prediction errors. Block8 has somewhat higher mean support Jaccard, but its subspaces are not uniformly well recovered. This separates fidelity, geometric stability and support stability rather than assuming one metric establishes all three.

### Control token-associated support

143 of512 rows belong to token identities with at least four occurrences. Hold the learned geometric matching fixed and shuffle the second model's support rows within exact token identity, five times. This preserves token-specific activation frequencies. On those143 rows, actual versus shuffled mean Jaccards are:

- Scalar48: .176 vs .096 at block1; .164 vs .089 at block2.
- Scalar64: .160 vs .104; .147 vs .098.
- Block8: .199 vs .139; .216 vs .152.
- Fixed shared: .067 vs .033; .073 vs .036.
- Learned shared: .037 vs .031; .032 vs .031.

Scalar/block support agreement contains context-dependent information beyond this simple exact-token frequency control. Learned shared gates again show little additional agreement. The repeated-token subset is small and selective; position, nearby lexical content, topic and activation magnitude remain uncontrolled. This is not evidence of task labels or smooth manifolds.

### Candidate selection and the ubiquitous-gate trap

At the exploratory thresholds overlap>=.5, Jaccard>=.3 and at least16 activations in both seeds, scalar48 has52/43 qualifying directions at blocks1/2; scalar64 has26/20; block8 has0/1; both shared families have0. Thresholds are descriptive, not validated definitions of a good feature.

An initial top-four ranking by overlap*Jaccard promoted some nearly ubiquitous block gates. Jaccard alone can be high when both gates are almost always on. Preserve that initial candidate file, then add an explicitly posthoc occupancy-adjusted ranking: overlap times positive Bernoulli support correlation (phi), requiring at least16 active AND16 inactive rows per seed. No causal task outcomes enter either ranking. This adjustment is a diagnostic response to occupancy, not independent confirmation.

The top scalar48 candidates remain highly matched. Block1 seed8351 units439,83,497,57 have overlaps.866–.907 and Jaccards.644–.761; block2 units176,432,415,140 have overlaps.834–.921 and Jaccards.632–.693. Their matched seed8352 units are saved. These are credible candidates for causal testing, not assigned concepts. Block2 block16 is the strongest adjusted block candidate (overlap.582, Jaccard.701); other block candidates have weaker geometric agreement. Test that limitation rather than calling every selected block a manifold.

### Twenty-hypothesis refresh

| # | Hypothesis | Current evidence / next discriminator |
|---|---|---|
|1|A subset of scalar directions is reproducible computational structure.|Strong matched subset exists; require selective, cross-seed causal effects.|
|2|Higher-dimensional blocks capture useful contextual states.|Some support stability, weaker subspace recovery; test within-block coordinates and effects.|
|3|Overlapping groups identify reusable semantic labels.|Current learned gates are unstable; selective atom reuse and held-out combinations remain absent.|
|4|Token identity alone explains support.|Within-token shuffle gaps weaken this for scalar/block models, not every confound.|
|5|Position or nearby lexical context explains remaining agreement.|Not removed by exact-token shuffle; match position and context controls in task tests.|
|6|Amplitude/scale creates apparent features.|Radius access explicit, but activity can depend on residual norm; matched-norm interventions needed.|
|7|Equivalent factorizations hide stable computation.|Geometric matching only partly resolves gauge; compare intervention effects across seeds directly.|
|8|Ubiquitous gates look stable without useful selectivity.|Observed candidate-ranking trap; use occupancy-adjusted support metrics.|
|9|Fixed group topology imposes arbitrary categories.|Fixed random groups are controls, not discovered ontology; compare data-informed groups later.|
|10|Membership concentration drives poor shared performance.|Confirmed85–102 covered atoms in learned variants; degree constraints separate collapse from utility.|
|11|Optimization, not architecture, limits groups.|Some plateaus and fewer active atoms; not fully excluded by longer training.|
|12|Response-weighted fitting identifies better axes.|Small gains, much less than lexical separation; requires downstream confirmation.|
|13|State-conditioned operators are better units than directions alone.|Baseline sensitivity and fixed-direction crossings support this locally.|
|14|RMS normalization is the main source of state dependence.|Local factor crossing points mainly to bilinear state, not RMS alone.|
|15|Later context-dependent maps erase local geometric advantages.|Downstream rank changes and common-response crossings support this possibility.|
|16|Output filtering masquerades as invariant computation.|Block56 softcap example demonstrates this failure mode.|
|17|Stronger sparsity regularizes a more reproducible dictionary.|Scalar48 more stable than64; distinguish regularization from occupancy/budget effects.|
|18|Local reconstruction/prediction is enough to rank interpretations.|Contradicted by downstream ranking changes; keep KL and causal-effect checks.|
|19|Task-specific features transfer across genuinely diverse computations.|Not established; competence-valid task panels are required.|
|20|Shared components are causally reused across held-out combinations.|Still untested for these exact-site features; decisive component-removal/transplant test remains.|

### Verification and next step

Ten site/model seed-pair comparisons completed. Orthonormal basis error1.33e-15; minimum group singular value.652, so rank-deficient QR padding is not driving overlap. All source hashes unchanged and metrics finite; no model forwards, reserved-test evaluations or new documents, natural-document pool26,112.

Freeze the occupancy-adjusted scalar48/block8 candidates, then evaluate their availability and cross-seed agreement on diverse tasks with native competence checks before causal interpretation. Keep agreement, copying, binding and factual-recall candidates from prior task work, and retain failures instead of substituting an easier task after intervention results. Test selected component removal/transplantation against matched-size controls and compare causal effects across matched seeds. This will determine whether stable directions/blocks are interpretable computational units rather than reproducible activation statistics. No semantic labels are assigned yet.
