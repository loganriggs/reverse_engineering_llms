# Data-selected context directions versus random subspaces

M102, completed 2026-10-29. This is a fixed null comparison for [M101's mixed-interaction construction](NATIVE_MIXED_INTERACTION_RESULTS.md), using MLP1 (zero-based layer 1).

## Question and theoretical prediction

For the bias-free bilinear write F, centered input u = z − mean and an orthogonal projector P, the candidate computes

`F_mixed(z) = F(z) − F((I−P)u)`.

It retains the mean and linear terms, quadratic terms inside the chosen subspace, and interactions between that subspace and its complement. Only complement–complement quadratic terms are removed. For an input change delta with centered midpoint h, its exact local change error is `J((I−P)h)(I−P)delta`. Thus a low-dimensional context subspace can modulate a much broader input without the entire computation living in that subspace.

If arbitrary small subspaces worked equally well, the earlier improvement would provide little evidence for useful data-dependent orientation. If PCA works better, orientation matters, although input variance could explain the advantage without discrete semantic units.

## Frozen comparison and verification

Three Gaussian-QR seeds (10201, 10202, 10203), each with nested ranks 4, 16, 64, were generated before loading the training mean. All nine conditions were retained. Each is compared with the same-rank Euclidean PCA candidate using the same mean, function, native weights and evaluation cohorts. This matches rank and functional construction, **not retained input variance or operator energy**. No seed or rank was selected using evaluation outcomes.

All evaluations and independent numerical audits passed: 352 natural-output vectors, 10,824 whole-task responses, and 2,772 source-group responses. The aggregate audit checked 4,752 quantities (maximum arithmetic discrepancy 7.11e-15). Matched comparisons contain 333 seed/rank/scope summaries; 2,536 native/shared-linear control rows replay exactly. Candidate preflight checked independent expanded predictions and actual forward replacement. The metric audits share the native Transformer implementation; they are not an independent Transformer implementation.

Whole-task exclusions and the historical source-group holdouts are preserved. Seven whole-task families remain in the full report, including weak native-task conditions; the qualified-five subset is also retained. All families, formats, seeds, ranks, intervention corners and aggregation methods are available in the JSON results.

## Results

At rank 64:

| Measure (lower is better) | Data PCA | Random seeds: minimum–maximum |
|---|---:|---:|
| Natural output KL | 0.02354 | 0.18865–0.19170 |
| Natural local relative squared write error | 0.17940 | 0.73341–0.73513 |
| Whole-task source causal error | 0.39620 | 1.99842–3.87671 |
| Whole-task final-position causal error | 0.28802 | 1.71168–1.81923 |
| Source-group source causal error | 0.34538 | 1.33456–1.50419 |
| Source-group final-position causal error | 0.77570 | 3.94397–12.47636 |

Causal rows use equal-family relative response error with both the background computation and intervention change reconstructed. Error 1 is the zero-response predictor, not a percentage of the model understood.

PCA has lower natural KL and local write error in all nine matched comparisons. It also has lower whole-task source and final causal errors in all nine, and lower source-group final error in all nine. Source-group source error improves in six of nine: **all three rank-4 random controls beat PCA** there (1.916–2.647 versus 3.453). These are aggregate results, not universal per-family or per-corner dominance.

All three rank-64 random candidates achieve 100% binary contrast accuracy on the source-group cohort, matching PCA and native, despite their large causal errors. Accuracy preservation therefore does not establish computational equivalence.

Training PCA64 captures 53.37% of centered input variance; random64 captures only 5.50–5.58%, close to the isotropic expectation 64/1152. This major variance difference prevents attributing the advantage specifically to semantic or causal alignment.

## What changes in our conceptual account

The M101 gains are not explained by choosing just any rank-matched context subspace. Data geometry matters. Together with M101, the evidence supports investigating context directions that modulate broader inputs, rather than assuming every useful unit must be a self-contained low-dimensional computation.

However, this identifies useful **subspaces**, not privileged axes or semantic units. The construction retains all 4,608 native products and their weights; it proves neither native-channel removal redundancy nor parameter compression. It also does not establish why training chose this implementation. Cancellation among approximation errors remains distinct from cancellation among native channels.

## Bounded next experiment

Hold the Euclidean PCA64 span fixed and rotate only its coordinates. The projector and complete candidate function must remain identical. Compare coordinate-level concentration and interaction signatures under the original axes and three fixed random rotations. Use algebraic toys and cached training inputs first; do not search rotations on task holdouts.

Two measurements must stay separate: (1) additive midpoint-Jacobian contributions to a fixed combined change, whose sum is invariant, and (2) actual one-coordinate finite interventions, whose sum need not equal the combined intervention because of cross-coordinate quadratic terms. This tests whether a claimed unit-level organization depends on arbitrary coordinates. It does not alter native channels or establish their cancellation.

Deliverable: a verified gauge-sensitivity diagnostic for evaluating featurizer units, with an explicit distinction between stable subspace computation and coordinate-dependent localization. Stop after the fixed rotations and toy/native arithmetic checks; any subsequent causal-role fitting requires a separate training-only protocol and untouched whole-task exclusions.

## Evidence

- [Matched results](validation/native_random_subspace_comparison.json)
- [Aggregate audit](validation/native_random_comparison_audit.json)
- [Training variance](validation/native_random_training_variance.json)
- [Natural audit](validation/native_random_natural_audit.json)
- [Whole-task audit](validation/native_random_factorial_audit.json)
- [Source-group audit](validation/native_random_source_factorial_audit.json)
