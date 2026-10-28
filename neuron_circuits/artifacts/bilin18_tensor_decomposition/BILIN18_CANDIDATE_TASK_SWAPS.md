## H841 — Candidate swaps are weak, but so is the entire early MLP task route

**Main result.** Frozen candidate swaps generally move task-answer contrasts very little. Crucially, swapping the entire normalized MLP input at sites1/2 also transfers little or negative donor preference. The negative candidate result therefore cannot cleanly distinguish a poor featurizer from a weak causal route at the tested location.

All16 shortlisted H839 components, both H836 training seeds, all80 pairs from five H840 competent families and both template splits were retained. We swapped donor-minus-recipient decoded component contributions into the final-position normalized MLP input, without altering the residual skip or other positions. This preserves native reconstruction error; it does not replace the whole input with a featurizer reconstruction. No renormalization is applied after the controlled module-input intervention.

Task swaps ran bidirectionally. Matched-norm Gaussian perturbations ran in both signs, with a frozen direction per candidate shared across seeds/pairs. Same-answer nuisance component donors and whole normalized-input donor controls were included. This is one ambient random direction per candidate, not a broad null over learned/task subspaces. All task endpoint pairs share the final token; some differ in length. Nuisance pairs can differ in both and are retained with explicit flags. They are not interchangeable causal specificity controls.

**Scale of effects.** We report donor-directed change in the native answer logit contrast. Gap fraction is mean change divided by the mean donor-versus-recipient native contrast gap, not accuracy and not mediation fraction. The full-input controls were:

|Site|Task|Donor-directed margin change|Fraction of native gap|
|---|---|---:|---:|
|1|agreement|-0.00451|-0.052%|
|1|capital_recall|0.00439|0.036%|
|1|entity_binding|0.00338|0.043%|
|1|sentiment|-0.01175|-0.425%|
|1|sequence_copy|-0.04067|-0.497%|
|2|agreement|0.00321|0.037%|
|2|capital_recall|0.01265|0.103%|
|2|entity_binding|0.00785|0.101%|
|2|sentiment|-0.03837|-1.387%|
|2|sequence_copy|-0.03579|-0.437%|

The largest positive candidate-family mean was site2 block candidate3 on copying:0.00537 contrast units, only0.066% of the native gap. Its excess over paired-sign random controls was0.00630 with descriptive95% bootstrap interval[-0.00189,0.01516]. Site2 block candidate2 on capital recall had a small consistent positive shift across both seeds/formats:0.00427 units,0.035% of the native gap; random-control excess0.00385, interval[0.00109,0.00727]. These are exploratory small effects among80 candidate-family comparisons; intervals are not multiplicity-corrected and do not establish a semantic unit. On copying, candidate2's apparent positive average includes a negative result in one transfer seed.

**Uncertainty and causal meaning.** Bootstrap resamples16 paired items/family after averaging seeds and donor directions. Final-output KL is saved separately from answer contrast: a component can perturb predictions without transferring the desired answer. Natural-text stability is still insufficient evidence for task-aligned computation. However, weak whole-MLP control results prevent attributing this solely to featurizer quality. The surrounding residual stream and other token positions may carry the relevant information.

**Next diagnostic.** Compare donor swaps of whole MLP output with whole block output at the same sites and position. The former should numerically reproduce full normalized-input donor swaps, validating the intervention implementation. The latter tests whether donor information transfers through the residual state while bypassing the weak early MLP contribution. Keep the attention value bus native and state that limitation. Do this before promoting tiny component effects or choosing new components to fit these tasks.

**Audits.**18,240 logical interventions,13,715 exactly zero component perturbations assigned exact identity outcomes,5,245 actual forwards including native and identity controls. Native normalized-input capture differs at most3.43e-6; identity replacement changes logits at most1.53e-5. All metrics finite; minimum KL0; source/checkpoint hashes unchanged; no reserved-test use. Natural-document count remains26,112. Previous goal turn was progress: it created the frozen diverse task screen used here.
