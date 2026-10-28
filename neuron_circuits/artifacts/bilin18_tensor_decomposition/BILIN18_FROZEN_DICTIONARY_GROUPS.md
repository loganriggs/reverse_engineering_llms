## H726 — Overlapping supports help modestly, but are not yet verified manifolds

We froze all six scalar dictionaries from H724 (three seeds at each of two causally relevant attention heads). Each dictionary has 128 reusable directions. Using only training activations, we inferred 128 groups containing four directions each; two groups are selected per example, and their union is decoded exactly once. This separates reusable directions from the groups selecting them, as proposed. It is a heuristic support-discovery experiment, not joint optimization of the proposed L0 objective.

Memberships use support correlations that replicate across two training halves. A second version subtracts current-token/position means before computing correlations. Three rewired controls per fit preserve every group's size and every atom's membership count. A disjoint partition control has fewer groups and memberships and is not capacity matched.

Validation relative write MSE (lower is better; averages over all three dictionary seeds):

| Support selection | Agreement head | Capital-recall head |
|---|---:|---:|
| Free scalar Top-8 | 0.3465 | 0.4420 |
| Raw coactivation groups | 0.4639 | 0.5429 |
| Degree-matched rewires of raw groups | 0.4885 | 0.5710 |
| Token/position-conditioned groups | 0.4796 | 0.5448 |
| Degree-matched rewires of conditioned groups | 0.5012 | 0.5719 |
| Random disjoint groups | 0.5380 | 0.6073 |

Inferred groups improve on degree-preserving random controls, but restricting supports loses substantial reconstruction fidelity relative to freely selected scalar coordinates. The two selected groups activate about 7.96 unique atoms out of eight nominal slots: simultaneous overlap is uncommon. Shared atoms can still be reused across different examples. This does not establish meaningful shared computation.

Across dictionary seeds, matched group-membership Jaccard overlap is 0.289/0.350 for raw groups and 0.250/0.332 for conditioned groups (agreement/capitals). Training-half overlap is 0.541/0.534 raw and 0.291/0.405 conditioned. These are descriptive, uncorrected for chance; atom alignment and group matching affect the metric. Conditioned inference also uses fewer eligible examples, so its difference from raw inference is not a clean test of lexical confounding.

Correctness: exact-union reconstruction matches the existing union implementation with zero error; rewiring preserves row and column degrees. Three dedicated unit tests pass: shared atoms decode once, second-gate selection uses uncovered energy, and atom permutations preserve behavior. All input hashes stayed unchanged and results were finite. No model forwards, new documents, or reserved-test evaluations were used.

Limitations: dictionary atoms are nonorthogonal, so greedy coefficient-energy gating does not optimize reconstruction error. Memberships are fixed-size and heuristic, and three random controls cannot support precise permutation significance. Budgets differ from scalar Top-8 in support-choice capacity; no description-length comparison has been performed. These results neither validate nor rule out better optimized overlapping featurizers.

Next: freeze these exact memberships and compare their task-transfer effects, nuisance sensitivity, output KL, and CE against scalar Top-8 and matched rewires. Only structures surviving reproducibility, functional fidelity, and selective intervention tests should proceed to manifold interpretation, hierarchy, or shared-computation claims.
