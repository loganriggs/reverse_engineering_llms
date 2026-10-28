## H684 — Residual blocks show reproducible structure beyond a scrambled reference

Use the H683 selected featurizers, with no new fitting. For every site/width, compare seed681 to seeds682 and683. Orthonormalize each decoder block and compute overlap as the mean squared principal cosine between subspaces. Hungarian assignment maximizes total overlap under a one-to-one block matching. This permits arbitrary rotations within blocks. Then compare matched binary activation support on4096 validation examples using Jaccard intersection/union.

The reference applies a deterministic signed permutation to the ambient coordinates of the second model's decoder, independently reoptimizes the block assignment, and evaluates support overlap under that assignment. It preserves each model's internal decoder geometry and original support frequencies while disrupting the shared ambient alignment. It is one reference per pair, not a significance test or a comprehensive null distribution.

Table reports pair-averaged means restricted to blocks selected on at least32validation examples in both models. Each method's eligible subset can differ because its matching differs. Full all-block metrics, assignments, scores and counts are retained in JSON; therefore these conditional means must not be treated as global recovery rates.

| Site | Width | Learned subspace overlap | Learned support Jaccard | Scrambled overlap | Scrambled support Jaccard |
|---|---:|---:|---:|---:|---:|
|value0|1|0.0554|0.0487|0.0106|0.0084|
|value0|4|0.0713|0.0737|0.0080|0.0081|
|value0|16|0.1279|0.1273|0.0172|0.0083|
|residual0|1|0.2498|0.1893|0.0108|0.0074|
|residual0|4|0.2483|0.2765|0.0081|0.0098|
|residual0|16|0.4218|0.3575|0.0168|0.0351|
|residual1|1|0.2448|0.1988|0.0109|0.0080|
|residual1|4|0.2298|0.2711|0.0081|0.0115|
|residual1|16|0.4142|0.3586|0.0170|0.0247|
|residual2|1|0.2820|0.2102|0.0107|0.0077|
|residual2|4|0.2388|0.2460|0.0081|0.0065|
|residual2|16|0.3831|0.2999|0.0169|0.0194|

Occupied width16 residual0/1 blocks reach about0.42 mean subspace overlap and0.36 support Jaccard, versus approximately0.017 and0.025–0.035 under the scrambled reference. The value stream has much weaker learned agreement (about0.128 for both at width16). Across residual sites, width16 has stronger conditional overlap/support stability than widths1/4, despite worse reconstruction. This is useful evidence that reconstruction ranking alone would miss a potentially interesting block organization.

The agreement is partial, not near-identical recovery. Wider subspaces also differ in chance overlap, occupancy, capacity allocation and ease of matching, so the numerical comparison is not a proof that16-dimensional blocks are the correct units. The reference addresses some matching effects, not all. Stable blocks could encode lexical categories, position, activation scale, or mixtures of unrelated factors. Subspace stability says nothing directly about whether the occupied points form a smooth manifold.

Residual0 and residual1 width16 blocks are now justified exploratory candidates for a small atlas, with value0 and width1 serving as useful contrasts. Next inspect robustly matched examples and internal coordinates while testing how much activity can be predicted by token identity/frequency and position. Compare same-token occurrences across contexts and held-out documents. A contextual residual after lexical controls would be stronger evidence for computational regime labels than a coherent list of token types. Use actual-example block swaps and downstream response prediction before assigning computational meaning. PCA and a conventional ReLU SAE comparison remain outstanding.

Audits:24seed-pair comparisons,36unchanged checkpoint files, all input/trainer/runner hashes unchanged, finite overlap metrics. SciPy1.18.1 was installed in the existing research virtual environment for exact assignment (initial attempt exited before analysis because SciPy was absent). No fitting, original-model forwards or test evaluations. FineWeb count unchanged26,112. H683 completed earlier this turn; broad research goal remains active and incomplete.
