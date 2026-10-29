# Operator compression transfers incompletely; mixed interactions are the next target

M100's fixed held-out comparison is complete. All nine candidates (three bases × context ranks4/16/64), the shared-linear baseline and exact native control were retained. The candidates use a shared linear term plus a quadratic computation restricted to projected context coordinates. They improve substantially over the shared-linear baseline, but are not faithful replacements.

## Behavior and causal fidelity

On natural held-out text, rank64 output KL was0.24550 for Euclidean input PCA,0.24185 for isotropic operator PCA, and0.25054 for task-direction-weighted operator PCA. The shared-linear baseline was0.48455; the native control was exactly zero. Operator-aware fitting therefore gives a modest, inconsistent retention advantage over the matched-rank Euclidean control.

The causal comparison below uses both the reconstructed background and reconstructed intervention change. Entries are equal-family relative response errors; lower is better and1 is the zero-response predictor. It is not a percentage of understanding or a fidelity threshold.

| Rank64 basis | Seven whole-task families: source | Seven whole-task families: final | Source-group holdouts: source | Source-group holdouts: final |
|---|---:|---:|---:|---:|
| Euclidean | 1.0825 | 1.1162 | 1.0568 | 1.4561 |
| Isotropic operator | 0.9945 | 1.1694 | 1.0134 | 1.6569 |
| Training-direction operator | 1.0721 | 1.2337 | 1.0192 | 1.7613 |

These are the prespecified highest-rank conditions, not selected winners; all ranks, families, formats and corners are saved in the comparison JSONs. At rank64, the five native-competent families retain95% contrast accuracy versus96.25% native accuracy, despite final-position causal errors1.1311/1.1626/1.2121. Source-group contrast accuracy is92.86% versus100% native.

Keeping the native background changes the picture. On all seven whole-task families, rank64 change-only source response error improves from0.7473 (Euclidean) to0.7099 (isotropic) and0.6851 (task-direction). Final errors are0.8420/0.7676/0.8122. Better predicted changes on a correct background do not imply a faithful full replacement. Scope and aggregation still matter: qualified-five source errors under full replacement are0.9789/0.8615/0.8287, unlike the all-seven values above.

## An exact local decomposition identifies a construction cost

Write the centered midpoint as h and the input change as delta. Let P be the fixed context projection. The native change is J(mu)delta+J(h)delta.

An anchor-only approximation uses J(mu)delta+J(Ph)delta. It preserves the entire change direction, but generally does not define a consistent function across all inputs.

Our executable candidate instead uses J(mu)delta+J(Ph)Pdelta. Its error separates exactly into

A = J((I−P)h)delta,

B = J(Ph)(I−P)delta,

native change − candidate change = A+B.

A is the anchor-only approximation error. B measures the additional effect of discarding change directions outside the selected context coordinates in this particular quadratic construction.

| Rank64 isotropic basis; pooled squared local error | Anchor only | Executable projected-quadratic function |
|---|---:|---:|
| Whole-task source | 0.3777 | 0.7701 |
| Whole-task final | 0.3582 | 0.7575 |
| Source-group source | 0.3590 | 0.7471 |
| Source-group final | 0.3752 | 0.7677 |

The same pattern appears for all three rank64 bases: anchor-only errors range0.3466–0.3900; consistent-function errors range0.7390–0.7791. The aggregate A/B cross term is positive in all twelve rank64 cohort/site/basis settings, so the added error term compounds rather than cancels the anchor error there. This statement does not assert the sign for every individual example or native channel.

**This is a limitation of the chosen function construction, not an inevitable price of integrability.** Restricting the quadratic function to projected inputs drops mixed interactions between retained and omitted coordinates as well as omitted-with-omitted interactions. A different consistent function can retain the mixed terms.

## Conceptual implication and next discriminating test

The evidence favors testing context coordinates as modulators of broader computation, rather than assuming all relevant input directions lie inside the same low-dimensional subspace. This is a functional hypothesis: it does not identify semantic gates, sparse blocks, native-channel redundancy or a computational DAG.

For u=z−mu, a fixed alternative is

Fmixed(z) = F(mu)+J(mu)u+F(Pu)+J(Pu)(I−P)u

= F(z)−F((I−P)u).

It retains interactions involving the selected coordinates and removes only omitted-with-omitted quadratic interactions. Its finite-change error is exactly J((I−P)h)(I−P)delta. It is integrable by construction. Whether removing that term helps depends on the data and cancellation; the algebra does not guarantee better causal fidelity.

M101 will test this alternative using the existing bases/ranks, with toy and native arithmetic checks before the fixed held-out comparison. No new fitting, rank selection or optimizer search is needed. This alternative retains more interaction structure and is not parameter-count matched to the M100 restricted-quadratic candidate; any gain must not be called a free compression improvement. The deliverable is evidence for or against a shared-context-coordinate mechanism suitable for a later featurizer.

## Verification

All11 executable candidates passed independent NumPy prediction, finite-change and guarded native-forward replacement checks on training data (24forwards). The natural evaluation used384forwards and saved352 full output vectors. Whole-task evaluation used8688forwards and independently audited10824 response vectors,3608 geometry rows and88 ineligible assignments. Source-group evaluation used2240forwards and independently audited2772 response vectors and924 geometry rows. Native control outputs replay exactly; causal control discrepancies from floating-point addition order were at most5.77e-12 absolute response error.

The projection decomposition covers all2448 candidate/eligible-nonidentity-target combinations. Independent NumPy polynomial replay agreed within1.08e-11 in vector norm and2.24e-8 in absolute energy metrics. The applied float32 candidate differences passed a predeclared rounding bound. All51408 scope-aggregation and comparison-arithmetic checks passed. The Transformer itself has not been independently reimplemented.

M100 is closed. These historical holdout results establish neither new-task discovery nor semantic manifolds, and no native neurons have been shown removable by this experiment.
