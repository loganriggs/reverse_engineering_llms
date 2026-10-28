## H895 — The failed leading PC misses available number information; larger subspaces are not consistently shared

The previous goal turn made progress: H894 completed disjoint-discovery recovery and exposed one failed first-PC number fit plus unstable compact contextual writes. This diagnostic asks whether input subspace geometry explains that failure. It uses the existing ten discovery captures only, with four noun pairs per fit, keeping the frozen H894 units unchanged. No new model forward, natural-document evaluation, or reserved test evaluation is performed.

### Number survives below the leading PC

For the failed old-transfer template-B fit, PC1 captures only1.03% of the singular/plural centroid-difference energy and has50% discovery number accuracy. PC2 captures22.07% with62.5% accuracy. PC3 captures76.01% and has100% accuracy. These are descriptive diagnostics of existing labeled discovery data, not a selected replacement unit or new confirmation test. Variance fractions for the first three PCs are32.47%,22.47%,20.69%: the highest-variance direction is not the best number direction.

Nearest-centroid decoding fitted in the failed fit's first two PCs transfers with100% accuracy to all four other template-B noun sets, despite capturing only21–29% of their number-centroid difference energy. Classification therefore gives a much weaker criterion than recovering most of the geometric contrast. A low-dimensional supervised readout may succeed without demonstrating a stable, complete computational representation.

### Input subspace overlap and transfer

For equal-rank orthonormal bases V and W, overlap below is the mean squared singular value of V W-transpose:1 means identical subspaces,0 means orthogonal. It is normalized by rank; adding unrelated directions can decrease it even when a shared direction remains. Number capture measures the fraction of a different noun set's centroid-difference energy projected into the source subspace. Transfer accuracy uses source-trained nearest centroids, so the decoder is supervised even though PCA is not.

Each template summary averages ten unordered geometry comparisons and twenty ordered transfer comparisons across disjoint noun sets. Endpoints and comparisons are dependent; these are descriptive means, not independent confirmation samples.

|Template|Rank|Mean normalized overlap|Number transfer accuracy|Target number-contrast capture|Target centered variance capture|
|---|---:|---:|---:|---:|---:|
|A|1|0.820|1.000|0.829|0.422|
|A|2|0.435|1.000|0.838|0.443|
|A|4|0.293|1.000|0.854|0.484|
|A|7|0.231|1.000|0.872|0.518|
|B|1|0.453|0.894|0.572|0.246|
|B|2|0.303|0.988|0.639|0.289|
|B|4|0.234|0.988|0.733|0.350|
|B|7|0.189|0.988|0.767|0.386|

Template A retains its stable one-dimensional number component: expanding from rank1 to7 increases average number capture only82.9%→87.2%, while normalized overlap falls0.820→0.231. Unnormalized shared energy (rank times overlap) grows0.820→1.614; thus this is not evidence that the whole shared subspace disappears, but additional dimensions are far less consistently aligned.

Template B improves average number transfer accuracy89.4%→98.75% at rank2, and number capture57.2%→76.7% by rank7. Yet normalized overlap falls0.453→0.189; even the full seven-dimensional discovery spans are not equivalent. Unnormalized shared energy grows0.453→1.326. The larger span partially recovers a recurring signal without establishing a single stable higher-dimensional manifold. These tiny discovery panels can also underestimate a genuine population subspace; instability is not proof that no manifold exists.

### Research implication and next experiment

This distinguishes missing information from an extraction rule that orders it poorly. The failed first PC does not imply absent number representation. Conversely, perfect number decoding does not validate a featurizer's manifold or conditional computation. Reconstruction-variance objectives may prioritize directions differently from causal relevance.

Do not replace the failed unit with its retrospectively identified PC3 and claim confirmation. Keep all ten H894 units frozen. Next evaluate them, together with exact component-removal controls, on the remaining384 locked natural documents. That measures prospective native-output fidelity and component influence without selecting the best fit. Input-subspace diagnostics do not answer whether output subspaces or compiled operators are stable; those and larger balanced discovery learning curves remain open. The original BSF-alternative comparison and manifold/computational interpretation objective remain active and incomplete.

### Verification

PCA bases are orthonormal to1.22e−15; first PCs reproduce the frozen H894 axes up to sign to3.55e−15. All source and frozen-unit hashes unchanged; all reported transfer energies finite. Zero new full-model forwards and zero reserved test evaluations. Natural-document pool remains26,112. Full per-PC, cross-template and ordered cross-set results are retained in the experiment JSON; the table summarizes within-template comparisons.
