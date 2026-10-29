# M57: early MLP causal coverage depends strongly on token position

**Source-position MLP1 write swaps have larger average effects than final-position swaps in every tested family.** For agreement, mean absolute answer-margin change is6.53 at changed input positions versus0.024 at the final position, and all24 source-position swaps reduce the recipient's intended-answer margin. This broadens the earlier color-only negative result: weak final-position effects do not establish weak early-MLP involvement overall.

## Frozen coverage, not a best-site search

The manifest was frozen before forwards:66 existing training pairs plus4 color training pairs; zero-based layers0/1/2; changed input-token positions, final position, and identity for every endpoint/layer. All native competence failures stay in the population. Existing lexical evaluation pairs and historical whole-family exclusions remain unused. No fits, candidate selection, or effect-size filtering.

At each selected address, replace the native product write with the opposite endpoint's native product write at the corresponding address, preserving Down bias. Changed-position patches require equal prompt length and compare token IDs by index. This is **address alignment**, not evidence of semantic alignment. The one unequal-length color pair is ineligible for that region at all three layers, yielding six explicitly recorded ineligible assignments; its final-position and identity conditions remain. Multiple changed positions are patched together. Conditions can overlap if a changed token is at the final position; no claim of disjoint regions is made.

## MLP1 family map

Source/final counts differ for color, so those two aggregates do not cover identical endpoint sets. Neither patch-vector energy nor position count is matched. Larger effects establish causal coverage under these interventions, not greater sensitivity per unit intervention or semantic purity.

| Family | Source/final endpoints | Source KL | Final KL | Source absolute margin change | Final absolute margin change | Source margin reductions |
|---|---:|---:|---:|---:|---:|---:|
| agreement | 24/24 | 0.75219 | 0.00071 | 6.5315 | 0.0244 | 24/24 |
| antonym | 6/6 | 0.05800 | 0.01229 | 0.2210 | 0.1628 | 4/6 |
| capital_recall | 4/4 | 0.01195 | 0.00014 | 0.3051 | 0.0146 | 3/4 |
| country_language | 4/4 | 0.00077 | 0.00002 | 0.1388 | 0.0133 | 3/4 |
| entity_binding | 12/12 | 0.00469 | 0.00012 | 0.2627 | 0.0243 | 6/12 |
| number_successor | 8/8 | 0.00394 | 0.00013 | 0.1772 | 0.0261 | 4/8 |
| object_color | 6/8 | 0.01689 | 0.00006 | 0.1763 | 0.0069 | 1/6 |
| past_tense | 4/4 | 0.14738 | 0.00062 | 1.8113 | 0.0339 | 4/4 |
| plural_noun | 6/6 | 0.28300 | 0.00195 | 1.6018 | 0.0522 | 5/6 |
| pronoun_gender | 32/32 | 0.00407 | 0.00002 | 0.1589 | 0.0083 | 27/32 |
| sentiment | 32/32 | 0.15182 | 0.00045 | 1.6494 | 0.0166 | 32/32 |

For color, only1/6 eligible source patches reduces the recipient answer margin; its larger response should not be mistaken for successful transfer of the donor's color answer. For pronoun gender, effects are small but directionally more consistent (27/32 reductions). Both effect size and direction matter, while neither independently validates a task-specific component.

## All three layers retained

Endpoint-weighted mean KL across all families, changed/final respectively:

- MLP0:0.02999 /0.000195
- MLP1:0.18780 /0.000883
- MLP2:0.02393 /0.001940

These pooled values reflect uneven family sizes, unmatched intervention magnitudes, and138source versus140final endpoints. They are descriptive, not a ranking of layer importance or proof of distinct computational strategies. Per-family/per-layer results and raw energies are preserved.

## What changes in the conceptual account

A plausible account is that early MLPs supply distributed, token-conditioned writes at input positions that downstream attention/readers use. The source-position effects support that account's causal-availability prediction. However, source swaps also change lexical information, so the evidence cannot distinguish an abstract agreement computation from an informative token-conditioned representation consumed elsewhere.

This separates three questions: where a signal is written, how its contribution is distributed across channels, and where the task-relevant transformation/readout occurs. It does not establish literal removal redundancy, cancellation, or meaningful featurizer manifolds. Those require their own matched native tests. The strong agreement effect is not grounds to start another agreement-only detour.

## Audit and concrete featurizer deliverable

974forwards produced1260 assignments:1254 evaluated vectors and6 explicit ineligible conditions. All420 identities retained. Independent NumPy audit rescores every final logit vector, all saved donor-minus-recipient deltas, address/split exclusions, and99summaries. Largest absolute discrepancy1.19e-7 includes large energy quantities. All8 previous M56 color MLP1 final-position results replay exactly. No independent transformer reimplementation is claimed.

Next apply **every eligible MLP1 target** from this frozen bank to all18 M37 initial/control/joint scalar/block/shared candidates. Use the same deltas and the same original inputs; compare each candidate's intervention response against the native response, including identity/null effects and all native errors. Report per-family metrics and equal-family macro summaries alongside pooled energies. Do not select families or positions from this map. This directly tests whether the existing featurizers preserve the newly mapped source-position causal information and feeds the diverse-task supervision loop without new fitting or held-out leakage.
