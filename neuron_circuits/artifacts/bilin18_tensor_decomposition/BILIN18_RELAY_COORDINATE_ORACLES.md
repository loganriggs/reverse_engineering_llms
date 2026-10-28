## H855 — Active spans can match the causal coordinate, but edit magnitude explains much of the apparent rescue

**Main result.** Every frozen donor/recipient active-union span tested can represent the independently frozen task coordinate. A constrained least-squares edit often restores most of the full relay's answer effect. However, scaling the ordinary Euclidean projection to the constrained edit's norm produces nearly the same rescue in many cases. Thus these results do not uniquely support a semantic-coordinate-aware objective over a simpler attenuation/calibration explanation.

**Exact diagnostic.** With orthogonal active-span projector P, true raw donor contrast delta and frozen unit task direction q:

`p = P delta`

`constrained = p + Pq * [qᵀ(delta-p)] / [qᵀPq]`.

This is the minimum Euclidean-error vector in the fixed span satisfying qᵀedit=qᵀdelta. It deliberately uses task supervision and the donor pair, so it is an oracle rather than a task-blind featurizer. UnlikeH854, these interventions operate directly in rawpreMLP4 coordinates without subsequent normalization/radius restoration. They need their own raw-reference comparison.

All10 fits and78H849 pairs retained, both directions. Controls: ordinary projection, other-task-coordinate constraint, Euclidean direction scaled to the own-constrained totalnorm, minimum-norm edit matching only the coordinate, direct task-rank1 edit, full rawdonor, identity. Same supports/subspace throughout. No unrepresentable constraints: smallest tested own/other-direction projected energy .0698, maximum active dictionary condition number4.98.

**Selected adjacent-cell examples.** Each entry is full relay mean effect preserved / individual-effect relativeRMSE. Both seeds and every task/gap cell are in the complete analysis; examples illustrate the norm-control issue rather than select a winning model.

|Task|Model|Euclidean|Coordinate constraint|Norm-matched Euclidean|
|---|---|---|---|---|
|agreement|scalar48_8351|50.2% / 0.513|112.2% / 0.154|108.9% / 0.137|
|agreement|scalar64_8352|51.9% / 0.489|107.3% / 0.097|104.1% / 0.081|
|agreement|block8_8351|57.1% / 0.463|110.3% / 0.138|106.0% / 0.128|
|agreement|shared_fixed_8351|29.6% / 0.712|100.3% / 0.164|95.8% / 0.190|
|agreement|shared_learned_8352|15.4% / 0.852|90.1% / 0.234|83.7% / 0.268|
|sentiment|scalar48_8351|59.6% / 0.423|103.2% / 0.116|98.1% / 0.123|
|sentiment|scalar64_8352|51.6% / 0.502|89.3% / 0.225|83.5% / 0.261|
|sentiment|block8_8351|60.7% / 0.417|100.1% / 0.167|94.3% / 0.173|
|sentiment|shared_fixed_8351|46.2% / 0.564|90.9% / 0.446|85.4% / 0.418|
|sentiment|shared_learned_8352|9.3% / 0.907|54.9% / 0.528|49.3% / 0.565|

Ratios above100% indicate overshoot, not better-than-perfect fidelity. The constrained edit's mean norm is about1.03–1.50 times the full donor contrast norm in these adjacent cells; the largest ratio across all cases is1.946. Ordinary projection norms are often only .49–.72 of the full contrast. These are not tiny perturbations, and norm increases matter. The norm-matched control itself uses the task-dependent constrained norm, so it is not yet a task-blind solution; H856 tests a direction-free norm rule.

Other-task constraints have little benefit, but that alone does not prove semantic specificity: they barely change the edit because the other coordinate has little native contrast. A strong comparison must control magnitude as well as which coordinate is enforced. The minimum-norm coordinate-only edits also work reasonably, but they introduce off-axis state within the span and do not establish that the model reads only one scalar.

**Interpretation.** H854's good full-dictionary result did not mean512 coordinates were necessary to reproduce the task effect: current active spans can carry the salient coordinate when coefficients are changed. Yet same-coordinate interventions still differ in output, especially learnedshared sentiment, and constrained effect error ranges widely. This is compatible with a compact signal distributed across learned atoms, attenuated by reconstruction/support choices, with downstream context and off-axis interactions. It does not establish a recovered semantic group or manifold.

**Audits.**8,268 full forwards. Float64 constraint error normalized by donor-contrast norm<=4.84e-16; span error<=2.39e-15; basis orthogonality<=4.45e-15. These audit the designed edits; model execution casts to float32 as usual. Identity logits exact; H849 full-swap margins replay within1.44e-5. All hashes unchanged and metrics finite; no reserved-test evaluation; natural-document count26,112 unchanged. All10fits and every seed/gap retained. Next isolate attenuation with a rule using only full-contrast norm and projected norm, without loading task directions, before deciding to redesign training around a supervised coordinate.
