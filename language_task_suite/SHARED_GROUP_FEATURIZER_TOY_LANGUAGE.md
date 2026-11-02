# Activation-only shared groups on a joint-rule toy language

The [joint-rule basis transfer test](JOINT_RULE_BASIS_TRANSFER.md) showed that a small response basis fitted on lexical and induction edits does not transfer well to previous-place edits. This experiment tests the user's proposed **shared dictionary plus overlapping group gates** directly on native model activations. The featurizer receives no rule labels or edit deltas during optimization. It does receive a *curated panel*: we use known edit labels to include lexical and induction contexts in fit/validation, then reserve the entire previous-place edit family for evaluation. The language model itself was trained on every rule. This is a featurizer-family holdout, not new-rule discovery from an uncurated corpus.

## Setup and controls

At the second block's pre-output-projection attention bond of the existing two-layer width-64 bilinear [joint-language checkpoint](surface_mixture_v2_soft_models/results.json), collect original and edited endpoint activations. Fitting uses 1,222 balanced lexical/induction pairs on one three-frame split, or 1,508 on the reverse split. Validation uses separate stream seeds and the same edit families. Tests use two new stream seeds (`20261113`, `20261114`) and the opposite three-frame set; **previous-place edits** are absent from all featurizer fit/validation pairs. A previous-place proposal can still be active in a lexical or induction context. The exact DGP oracle and complete native model remain fixed.

The [trainer](train_surface_v2_shared_groups.py) imports the already unit-tested `SharedGroups` implementation from `neuron_circuits/scripts/feature_structures/models.py` in the companion workspace. All learned models start with the same full PCA decoder/encoder initialization, train for 1,200 steps with identical minibatch indices, and reconstruct normalized native bond activations. The scalar normalization and train mean are fit only on the training panel. Test intervention transfer reconstructs original and edited endpoints **separately**, subtracts those reconstructions, and patches that difference into the original model. This matters because a gated encoder is nonlinear.

The group models have 64 dictionary atoms, 16 possible groups, two active groups per example, and four atoms per group. An atom in both selected groups is decoded once. The learned version trains group membership; the disjoint control fixes four nonoverlapping atoms per group. The shared model has 10,384 trainable parameters, fixed disjoint 9,360, scalar Top-6/Top-8 8,320. PCA-8 is a parameter-free fitted subspace. The two groups can activate at most eight distinct atoms, and overlap can lower that number. Scalar Top-6 and Top-8 bracket the observed active-atom count.

| Method | Active atoms, forward/reverse | Test reconstruction MSE, forward/reverse | Held-out previous-place patch-to-full L1, forward/reverse |
| --- | ---: | ---: | ---: |
| PCA-8 | 8/8 | .509/.644 | .281/.382 |
| Scalar Top-6 | 6/6 | .183/.303 | .197/.181 |
| **Scalar Top-8** | **8/8** | **.157/.230** | **.171/.169** |
| Fixed disjoint groups | 8/8 | .381/.560 | .367/.432 |
| Learned shared groups | 6.85/6.23 | .449/.666 | .431/.394 |
| Shared groups + coverage cost | 7.85/7.80 | .463/.644 | .435/.439 |

MSE is in train-scaled activation coordinates. Patch-to-full L1 compares the edited probability effect after the restricted featurizer patch with the effect of a complete 64-dimensional patch **at this same bond**. Lower is better; it is not the model-to-oracle error. The complete local patch approximates the complete model's previous-place effect within L1 **.032** on the forward panel, so this is a causally relevant site for that rule. The result is similarly negative for induction: scalar Top-8 patch-to-full L1 is **.209/.187**, learned shared groups **.375/.316**, and coverage-constrained shared groups **.344/.386**. The lexical edit has other major routes, so its local-patch score is not a complete lexical-circuit test.

The unconstrained learned memberships use only **20/18 of 64** distinct atoms on forward/reverse splits, though all 16 group gates receive nonzero use. A coverage cost targeting at least 48 distinct atoms raises coverage to **47/46** and retains **13/14** atoms shared between groups. Its effective active count rises near eight, but reconstruction and previous-place causal fidelity remain far below scalar Top-8. Thus support collapse alone does not explain the group model's disadvantage. The learned gate changes on **1.18/.93** groups for previous-place edits and **1.15/1.00** for lexical edits; this does not reveal a selective held-out-rule gate. Oracle-stable nuisance-adjective edits elicit small model effects and small featurizer patches, but that selectivity does not rescue the previous-place result.

This is a **negative result for this site, objective, budget, and two featurizer fits**. Group selection is much more constrained than arbitrary scalar Top-8 support, even though the learned group model has more parameters. The training pairs were curated by known labels, so this is not task-blind discovery; the test panel is fresh but both results use one fixed language-model seed. It does not refute overlapping groups, hierarchical gates or DAG computation models generally. A better next experiment would constrain membership and gate learning with causal objectives or use a component site whose native computation is demonstrably low-dimensional, while keeping scalar Top-K and the full local patch as controls. Further plain reconstruction-only tuning of this site has a stopping point unless it beats the scalar baseline or reveals a stable, selective new group.

Artifacts: [forward baseline](surface_mixture_v2_shared_groups_forward_v1/results.json), [reverse baseline](surface_mixture_v2_shared_groups_reverse_v1/results.json), [forward coverage](surface_mixture_v2_shared_groups_coverage1_forward/results.json), and [reverse coverage](surface_mixture_v2_shared_groups_coverage1_reverse/results.json). Each directory also contains the frozen featurizer weights. The imported `SharedGroups` exact union, gradient, and shared-atom behavior pass the companion workspace's `tests/test_feature_structures.py` (10 tests).

## Paired activation-change objective

A follow-up trains every learned method with the **same paired objective** on the same lexical/induction fit panel:

\[
L = \operatorname{MSE}(\hat z,z)
  + \lambda\operatorname{MSE}(\hat z_1-\hat z_0,z_1-z_0).
\]

Here `z0` and `z1` are native bond activations before/after one supported edit. Both endpoints are reconstructed through the nonlinear featurizer before subtraction. The shared-group method also receives the same coverage cost as in the prior correction. The zero-change control was rerun under the new validation-objective selection rule; all methods share paired minibatch indices within each nonzero-`λ` run. The previous-place family remains excluded from fitting and validation. This supervises *local activation changes*, not downstream probabilities or exact-oracle effects.

| Change weight `λ` | Shared groups: previous-place patch-to-full L1, forward/reverse | Disjoint groups | Scalar Top-6 | Scalar Top-8 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | .435/.439 | .367/.432 | .197/.181 | **.171/.169** |
| 1 | .403/.413 | .394/.400 | .196/.171 | **.165/.145** |
| 5 | .434/.481 | .436/.441 | .170/.183 | **.153/.152** |

The scalar result also wins **test activation-change MSE**. At `λ=1`, scalar Top-8 gets **.077/.133** versus shared groups **.240/.343** in train-scaled coordinates. The learned groups' distinct-atom coverage remains **42/41** under `λ=1` and **40/42** under `λ=5`, with about 7.3–7.6 active atoms per example. The shared model's nuisance-adjective patch TV falls at `λ=1` to **.0065/.0098** versus scalar Top-8 **.0109/.0156**, but its previous-place effect is also attenuated and much less faithful to the full local patch. This is not a selective new-rule group.

The bounded result is that local change supervision helps this scalar dictionary more than the current grouped dictionaries. It does **not** test a downstream KL/CE objective, intervention-aware group routing, a different site, or a larger naturalistic DGP. Further sweeps of this same reconstruction/change loss on this same panel have a clear stopping point. Saved [zero-change forward](surface_mixture_v2_shared_groups_change0_forward/results.json), [zero-change reverse](surface_mixture_v2_shared_groups_change0_reverse/results.json), [weight-1 forward](surface_mixture_v2_shared_groups_change1_forward/results.json), [weight-1 reverse](surface_mixture_v2_shared_groups_change1_reverse/results.json), [weight-5 forward](surface_mixture_v2_shared_groups_change5_forward/results.json), and [weight-5 reverse](surface_mixture_v2_shared_groups_change5_reverse/results.json) files include model weights, reconstruction, change, causal-patch and nuisance metrics.
