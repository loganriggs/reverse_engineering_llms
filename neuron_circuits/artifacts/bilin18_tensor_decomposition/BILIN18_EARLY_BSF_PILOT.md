## H682 — Early block-sparse pilot: multidimensional activity, no reconstruction advantage yet

Train36 signed block-TopK featurizers: four early sites×block widths1,4,16×three seeds. All use2304 decoder coordinates,32 active scalar coordinates per example, train-only centering/global scaling, unit decoder-column norms, Adam learning rate0.0003, batch512 and1000 steps. Validation every250 steps chooses checkpoints. Same seeded minibatch streams and initial flattened weights across widths. No diagnostic tasks or labels are used.

The width1 comparator is a signed absolute-TopK sparse dictionary, not a conventional nonnegative ReLU SAE. Wider blocks select8 or2 groups rather than32 scalar features; the same scalar budget does not imply the same flexibility or support-code cost. Decoder columns are individually normalized, not block-orthogonal. This is a custom baseline implementation of block-L2 gating, not a reproduction claim for a published BSF method.

| Site | Block width | Mean validation relative MSE | Mean training relative MSE | Validation dead blocks across seeds | Median effective rank, seed mean |
|---|---:|---:|---:|---|---:|
|value0|1|0.2526|0.1847|[0, 0, 0]|1.00|
|value0|4|0.2940|0.2255|[0, 0, 0]|2.77|
|value0|16|0.3113|0.2443|[0, 0, 0]|5.42|
|residual0|1|0.2508|0.2126|[0, 0, 0]|1.00|
|residual0|4|0.2716|0.2421|[1, 3, 1]|2.25|
|residual0|16|0.2966|0.2753|[15, 15, 12]|5.53|
|residual1|1|0.3186|0.2614|[0, 0, 0]|1.00|
|residual1|4|0.3466|0.2991|[1, 3, 1]|2.53|
|residual1|16|0.3721|0.3333|[9, 9, 9]|6.50|
|residual2|1|0.3615|0.2975|[0, 0, 0]|1.00|
|residual2|4|0.3949|0.3403|[0, 0, 0]|2.66|
|residual2|16|0.4289|0.3811|[6, 10, 8]|7.03|

Relative MSE divides squared reconstruction error by centered/scaled activation energy. Dead means never selected on4096 validation vectors, not provably inactive on the full distribution. Effective rank is the covariance participation ratio (trace squared / sum of squared eigenvalues) of coordinates conditional on block selection, restricted to blocks with at least32 validation selections. It measures dimensional usage, not smoothness, connectedness, semantic unity or causal validity.

The one-dimensional baseline reconstructs better at every site in this fixed-compute comparison. Width4 blocks use roughly2.2–2.8 effective dimensions; width16 blocks roughly5.4–7.0. Some width16 residual blocks are unused on validation. Multidimensional codes therefore exist, but this does not yet establish good manifolds or an advantage of the block organization. A block could mix independent factors or unrelated lexical identities.

Every run selects step1000, the final available checkpoint, and validation error improves substantially from250 to1000. We cannot claim convergence or attribute the reconstruction ranking solely to representational suitability. Training duration and optimization difficulty remain live confounds. Next extend the predeclared comparison to a longer fixed schedule, save optimizer states for resumability, and measure whether the ranking and block usage stabilize. The current saved models contain selected weights and training statistics but not optimizer state; extending them requires either an explicitly documented optimizer restart or deterministic replay from the original seed, not a claim of exact resume.

Then compare learned decoder subspaces across seeds, allowing rotations within a block; compare activity support and conditional geometry as well, since stable span alone is insufficient. Include matched null alignment and PCA references. Only after the fitting protocol is frozen should the reserved test split be evaluated. Causal usefulness and candidate regime labels remain subsequent tests; no interpretable manifold has yet been established.

Audits:36 saved models,1000optimizer steps each, finite validation results and unchanged activation/manifest/runner hashes. Mean active-block count is monitored in each record (32/r by construction). Zero model forwards, zero test evaluations, no new FineWeb documents. The complete document-separated test split remains unused. Prior turn completed data collection. Broad research goal active and incomplete.
