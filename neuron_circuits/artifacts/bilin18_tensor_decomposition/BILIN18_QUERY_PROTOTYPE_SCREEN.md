## H930 — Task-blind query prototypes recover useful context groups, with uneven stability

H929 required supplied context labels to predict coefficients well. H930 fits groups from query activations without domain, task, template or correctness labels. It screens whether such groups are stable and predict persistent-value writes; downstream causal fidelity is not yet tested in this experiment.

### Implementation and splits

At each fixed reader, concatenate the two normalized final-token query vectors before rotary position encoding (256 coordinates). Fit independent k-means models with1,2,4,8 prototypes and three predetermined seeds, on the96 discovery examples. No evaluation examples enter prototype or coefficient fitting. For each prototype, average discovery routing coefficients by distance from the final token. Gate evaluation examples by their nearest query prototype and predict their coefficient vector. Uncovered distances would fall back to the discovery positional mean; no fallback was needed.

Queries are captured before rotary encoding to avoid directly clustering the rotary phase. They can still encode position, final-token identity, syntax and lexical context. Task-blind fitting does not imply task-pure or semantic groups. Clustering uses a fixed pretrained coordinate system and Euclidean distance; no gauge-invariant representation is claimed.

The96 evaluation examples use familiar transfer lexical pairs across agreement, language, binding and copying. The coefficient target is native routing and the reconstruction target is the unprojected persistent head write. This is not feature discovery from reconstruction alone: causal validation remains required.

### Local write prediction and stability

Write error is pooled RMS reconstruction error divided by the RMS native persistent write; zeroing has ratio1. Adjusted Rand index (ARI) measures agreement of held-out cluster assignments across seeds, ignoring arbitrary cluster numbering;1 means identical partitions. The one-cluster ARI is trivially1.

|Reader|Prototypes|Write-error range across seeds|Held-out ARI range|
|---|---:|---:|---:|
|L8H3|1|0.623–0.623|1.000–1.000|
|L8H3|2|0.329–0.329|1.000–1.000|
|L8H3|4|0.253–0.297|0.384–0.813|
|L8H3|8|0.208–0.219|0.797–0.860|
|L13H0|1|0.821–0.821|1.000–1.000|
|L13H0|2|0.623–0.770|0.412–1.000|
|L13H0|4|0.209–0.209|1.000–1.000|
|L13H0|8|0.176–0.196|0.590–0.914|
|L14H4|1|0.960–0.960|1.000–1.000|
|L14H4|2|0.764–0.873|0.092–0.412|
|L14H4|4|0.795–0.796|0.635–1.000|
|L14H4|8|0.543–0.558|0.859–0.907|

L13H0 with four prototypes is a particularly clear result: all seeds give identical held-out partitions and write error0.209, compared with0.821 for the pooled positional rule and0.166 for the supplied-context mean. Retrospective discovery composition separates language task and binding task into their own groups. The remaining two groups mix agreement task and parts of agreement copying with other copying contexts. This is useful structure, not a perfect unsupervised task ontology.

L8H3 two-prototype partitions are also stable, but finer partitions vary substantially at four prototypes. Eight prototypes improve its write error to0.208–0.219 versus supplied-context0.189. L14H4 improves to0.543–0.558 at eight prototypes, near the supplied-context0.527 but leaving substantial local write error. More clusters do not automatically establish meaningful manifolds, and similar errors can coexist with different partitions.

### Interpretation and next test

Small task-blind query codebooks can identify context distinctions that improve coefficient prediction. The useful representation may consist of lexical carriers plus a small context state selecting how to route them. However, prototype stability is measured across initialization seeds on the same discovery sample, not independently sampled datasets. Template and final-token confounds remain. These groups are discrete prototypes, not verified continuous manifolds.

Next test prototype gates during live intervention rollouts, so earlier edits can change later queries and their assignments. Evaluate predetermined4- and8-prototype models across all three seeds, retaining the1-prototype baseline and H929 context-mean reference. Measure native KL, full donor-value intervention KL and answer-effect fidelity. Do not select the best seed or prototype count after outcomes. Stability alone cannot justify replacing the model's computation; the intervention test must show what the groups preserve. After that, assess new templates and disjoint fit samples before semantic labeling.

### Verification

192 native forwards. Native-cache max logit error2.48e-5; captured routing replay error4.25e-7. Signed query/key product times values reproduces head outputs within4.58e-5 absolute error. Discovery/evaluation indices disjoint; prototype initialization seeded; nonempty clusters and finite centers asserted. Source hashes unchanged, all results finite. Pre-rotary query captures, prototype centers, coefficient means, assignments, retrospective composition and local errors saved. No reserved test or added natural documents; pool remains26,112. Broad goal active.
