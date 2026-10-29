# M58: dominant-effect fidelity hides uneven task coverage

**Source-position causal fidelity is strong under pooled native-response energy but much weaker when families receive equal weight.** Agreement contributes74.1% of the source response energy and sentiment15.9%; together90.0%. Scalar seed71 KL-only has pooled normalized error0.155 versus equal-family error1.039. Zero response scores1. This is not contradictory: the two aggregates answer different questions.

## Fixed comparison and scope

All18 M37 initial/KL/KL+response states were tested on exactly the same M57 MLP1 deltas:70 training pairs,140 recipients,138 changed-position targets,140 final-position targets,140 identities. The two ineligible changed-position assignments remain explicit for every candidate. No task, native-error, effect-size, or candidate filtering and no new fitting.

Each model reconstructs the original entire MLP1 product write. The fixed native delta is added externally at the selected addresses, and the candidate's response is measured against its own reconstructed intact baseline. This tests downstream response fidelity in the reconstructed context. It does **not** test whether the encoder discovers the intervention as a semantic group, or whether native channels can be removed redundantly. The common-target construction allows candidate comparisons but active union sizes remain unequal.

## Final states, with both aggregations

Errors sum native-probability-weighted squared final logprob response errors. Pooled error divides total error by total native response energy. Equal-family error averages each family's normalized error, so small-energy families receive the same weight as large-energy ones. Small denominators can make ratios unstable; all raw energies and per-family values are saved. Neither aggregation is the sole correct measure. Every identity family has zero target energy, is reported undefined for normalized error, and has exactly zero absolute error.

| Representation | Seed | Objective | Source pooled | Source family mean | Final pooled | Final family mean |
|---|---:|---|---:|---:|---:|---:|
| scalar | 71 | KL | 0.155 | 1.039 | 0.842 | 1.214 |
| scalar | 71 | KL+response | 0.140 | 1.257 | 0.650 | 1.181 |
| block | 71 | KL | 0.157 | 1.666 | 1.507 | 1.433 |
| block | 71 | KL+response | 0.146 | 1.402 | 1.077 | 1.248 |
| shared_fixed | 71 | KL | 0.257 | 1.806 | 1.036 | 2.010 |
| shared_fixed | 71 | KL+response | 0.216 | 2.005 | 1.229 | 1.722 |
| scalar | 72 | KL | 0.147 | 1.098 | 1.161 | 0.940 |
| scalar | 72 | KL+response | 0.132 | 1.386 | 1.205 | 0.916 |
| block | 72 | KL | 0.179 | 1.883 | 1.205 | 1.316 |
| block | 72 | KL+response | 0.172 | 1.374 | 1.115 | 1.239 |
| shared_fixed | 72 | KL | 0.288 | 1.643 | 1.357 | 1.559 |
| shared_fixed | 72 | KL+response | 0.258 | 1.978 | 1.101 | 1.401 |

All six matched architecture/seed comparisons improve pooled source error with added causal supervision, but only the two block comparisons improve equal-family source error. Scalar/shared source-family averages worsen despite better pooled scores. This establishes evaluation heterogeneity; it does not prove that the original training loss used the same weighting or caused the pattern.

For scalar71 KL-only, source family errors range from agreement0.120 and sentiment0.114 to entity_binding3.009 and country_language3.126. The last two have only0.17% and0.01% of target energy respectively. Strong signals dominate pooled success; weak signals still expose false or inaccurate responses. Initial states and all family scores remain available, not just the final-state table.

## Conceptual account updated

These featurizers can preserve some large, distributed input-position causal effects without preserving every behavior or revealing reusable semantic units. Early MLP writes can carry token-conditioned information that downstream layers use; fidelity to dominant writes does not establish a universal computational decomposition. Distribution across channels, clustering/token geometry, and downstream use remain separate axes.

A claim such as “we understand X% of MLP1” would be unjustified from this score. It depends on which interventions, tasks, positions, and weights define the target, and these are familiar training contexts. The result is a benchmark-specific fidelity measure, not a percentage of mechanistic understanding or removal redundancy.

## Verification

7942forwards;140native baseline and418native target vectors replay exactly. There are7560candidate assignment records (7524evaluated,36ineligible) and2520intact baselines. Every saved final vector is independently rescored in NumPy; all7524delta hashes,648summaries and54macros are checked. Maximum metric discrepancy1.24e-14. The audit does not independently implement transformer forwards. All outputs are archived per recipient in hash-linked local files.

## Next bounded transfer check, before fitting

Apply the same MLP1 donor/region definitions to the14 pre-existing source-group evaluation pairs from `diverse_supervision_split.json`, retaining every candidate and native failure. Freeze the inventory first, and report families with no held-out source groups as unavailable rather than filling them with training examples. This is familiar lexical/source-group development transfer, **not** fresh whole-task validation or task discovery. Existing whole-task training exclusions remain untouched.

The check is necessary before treating the training-panel causal-supervision gains as transferable. It is not a task-selection step and does not authorize a loss-weight sweep. After the fixed holdout check, reconcile training-versus-transfer and inspect the actual objective calibration before specifying any new training treatment. Preserve pooled, equal-family and raw-energy views together; do not optimize one headline against these evaluation outcomes.
