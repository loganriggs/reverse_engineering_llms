# M47: context pairing matters to native outputs, with task-dependent transfer

M46 established within-token predictive covariance. M47 tests its downstream relevance by comparing correctly paired missing-half predictions with a fixed same-token context permutation. All six maps and token means remain frozen.

## Fixed intervention

For each eligible domain/source-split/token group from M46, sort distinct prefixes by SHA256 of their token IDs and cycle each recipient to the next prefix. This gives a deterministic bijection with no self donors; no seed or outcome search. Eligible positions receive either their own frozen missing-half prediction or the donor prediction, plus their own retained native half and bias. Ineligible positions retain the complete native write in both arms. All downstream computation remains live.

Eligibility is identical to M46:2610unique prefixes across the three cohorts. The permutation preserves each token’s prediction marginal over unique prefixes. Full-input evaluation reweights repeated prefixes, so occurrence-weighted marginals need not match. Donor states can come from later positions or other sequences: this is an oracle intervention control, not a deployable predictor or an inference algorithm.

## Results

KL is native||edited over full output distributions, averaged over positions, inputs and the two complementary masks. Local error is summed write error over modified positions. Ratios below1 favor correct pairing.

| Cohort | Layer | Paired KL | Permuted KL | KL ratio | Local-error ratio | Paired wins / cases |
|---|---|---:|---:|---:|---:|---:|
| task train | MLP0 | 0.00391 | 0.00490 | 0.7973 | 0.9012 | 176/264 |
| task train | MLP1 | 0.00891 | 0.00927 | 0.9612 | 0.9365 | 168/264 |
| task train | MLP2 | 0.00883 | 0.00859 | 1.0275 | 0.9628 | 126/264 |
| task evaluation | MLP0 | 0.00155 | 0.00201 | 0.7743 | 0.8977 | 48/56 |
| task evaluation | MLP1 | 0.00563 | 0.00528 | 1.0658 | 0.9391 | 23/56 |
| task evaluation | MLP2 | 0.00584 | 0.00596 | 0.9791 | 0.9506 | 35/56 |
| natural evaluation | MLP0 | 0.00593 | 0.00892 | 0.6649 | 0.7679 | 63/64 |
| natural evaluation | MLP1 | 0.01555 | 0.01945 | 0.7995 | 0.7628 | 60/64 |
| natural evaluation | MLP2 | 0.01251 | 0.01545 | 0.8100 | 0.8918 | 62/64 |

Natural paired output KL improves33.5%,20.0%,19.0% atMLP0/1/2. Both mask directions improve in every layer;63/64,60/64,62/64document–mask cases respectively favor pairing. These counts are dependent observations, not confidence intervals.

Task transfer is mixed despite better local reconstruction in every aggregate. MLP2 task-training output KL is2.7%worse under correct pairing; MLP1 task-evaluation is6.6%worse, with the latter reversal concentrated in one mask direction. Correct context alignment therefore does not guarantee better output fidelity for every distribution or task. The task splits are familiar development strata; neither supplied data for fitting these maps.

## Conceptual conclusion

There is shared contextual information between fixed native channel halves that matters to downstream predictions on natural text. Token identity alone cannot explain the M46 contrast, and M47 shows that its correct pairing is behaviorally relevant under this intervention. This is stronger evidence than raw coactivation or reconstruction alone.

It is still weaker than interchangeable computation. Only a selected eligible subset is replaced; paired outputs remain different from native; donors are oracle states; tasks show reversals. The experiment does not identify which context variables matter, distinguish copied common input from a reused operation, or show that the learned mapping corresponds to native semantic units. Position, topic and syntactic state remain candidates. No claim of general removal redundancy, optimal compression, or new-task discovery follows.

Absolute KL must not be compared directly with M42/M45 as an architecture improvement: those stages replaced every position, while M47 leaves all ineligible positions native. The informative contrast here is paired versus permuted under matched eligibility.

## Research direction and stopping criterion

Close the bounded native-half context-sharing sequence. Its deliverable is an audited token-conditioned, context-pairing causal control that featurizer groups should pass before being called reusable. Return to the frozen shared-dictionary candidates: distinguish exclusive group contributions from shared atoms and require native effect validation with token/context controls. Do not extend donor, amplitude or map searches on this panel. Keep reconstruction, ordinary prediction, causal response, semantic interpretation and new-task discovery separate.

## Audit

2,496forwards/2,304intervention rows. Independent audit reconstructs the entire deterministic donor permutation and eligibility, checks bijection/no fixed points, inventories and summaries, and verifies12sampled NumPy writes exactly. Ineligible writes remain bitwise native in sampled captures. Maximum sampled output metric difference5.50e-16. No independent transformer implementation.

[Protocol](validation/native_half_pairing_native_protocol.json) · [Native results](validation/native_half_pairing_outputs.json) · [Audit](validation/native_half_pairing_outputs_audit.json)
