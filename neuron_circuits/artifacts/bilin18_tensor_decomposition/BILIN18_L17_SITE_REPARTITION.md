

## H309 — Site-level repartition: document-context overlap has a small effect

Use exactly the H308 cached response vectors, source hash verified. Three seeded permutations of4096 positions into four1024-position blocks, same permutation for all18 groups. Specialist output PCA ranks32/128; all within/cross pairs. Unlike H308, positions of one document may occur in multiple blocks. This intentionally shares document context across fitting and evaluation; it is a dependence diagnostic, not valid fresh-document generalization. Every individual response vector remains in one block only.

| Rank | Grouping | Within capture mean | Cross capture mean | Gap mean |
|---|---|---:|---:|---:|
|32|document|34.70%|7.91%|26.79%|
|32|site|34.35%|8.29%|26.06%|
|128|document|68.64%|23.98%|44.66%|
|128|site|68.29%|24.62%|43.67%|

Across directed block pairs, 57.4% of evaluation sites have another site from the same document in the fitting block on average. This confirms substantial document overlap despite the small transfer gain.

Locked document-dependence gate FAILS: requires>=5pp mean cross-capture gain at BOTH ranks relative to whole-document repartitions. Observed gains are only.383pp rank32 and.642pp rank128. The site-level gap remains about26.06pp/43.67pp. Three repetitions give descriptive reproducibility, not confidence intervals or a calibrated significance test.

**Inference.** The original block boundaries and keeping four positions together are not sufficient explanations for the large PCA fitting/transfer gap in these controls. The small overlap benefit does not imply there is no document dependence: sampled positions can differ substantially within a document, and this manipulation tests only dependence exploitable by these global second-moment directions. Token/context-specific computations, heavy-tailed response magnitudes and high-dimensional sample noise remain plausible. No semantic feature ontology or irreducible rank has been established.

**Audit.** Every permutation uses all4096 site indices once; each block has1024 positions. All result arrays are4×4×18 with captures in[0,1]. Document overlap computed from site-index integer division by4, consistent with the saved1024×4 response layout. Source hash checked before analysis. No native model execution or weights changed; all jobs terminal.

**Next discriminator.** Measure response-norm concentration and repeat covariance transfer after giving each response equal norm, while reporting the changed metric explicitly. This separates amplitude-driven sample concentration from broad directional structure. Compare raw and normalized-direction results on identical partitions, and evaluate any learned directions back in the raw-energy metric as a separate result. A normalization effect is not automatically better causal reconstruction. Broad research goal remains active.
