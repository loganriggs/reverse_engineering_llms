## H731 — Learned memberships help agreement transfer, but do not establish group-specific computation

We froze every H730 membership set and evaluated all dictionary seeds, both initializations, initial and learned memberships, and three degree-preserving rewires per learned fit. Identity and scalar Top-8 controls were rerun. All groups use the same greedy-plus-best-response inference from H730.

Task intervention adds F(donor head) minus F(receiver head) to the native receiver head, allowing downstream computation to run. Natural-context evaluation instead replaces the head by F(native head). The two readouts answer different questions: retained counterfactual information versus output-distribution fidelity. This is the reused agreement/capital panel, not a new diverse-task confirmation.

| Representation | Agreement task-effect retention | Capitals task-effect retention | Agreement output KL | Capitals output KL |
|---|---:|---:|---:|---:|
| Scalar Top-8 | 94.0% | 67.1% | 0.000526 | 0.001949 |
| Initial correlation groups | 90.2% | 53.8% | 0.000812 | 0.002113 |
| Learned correlation groups | 92.7% | 53.8% | 0.000821 | 0.002127 |
| Degree-matched rewires of learned correlation groups | 90.0% | 48.5% | 0.000767 | 0.002775 |
| Initial random groups | 89.9% | 50.5% | 0.000827 | 0.002929 |
| Learned random groups | 94.6% | 50.8% | 0.000774 | 0.003004 |
| Degree-matched rewires of learned random groups | 94.9% | 49.9% | 0.000773 | 0.002800 |

Task retention is mean donor-directed margin change divided by exact-head intervention's mean change on eight transfer pairs, not accuracy. Values average three dictionary seeds; rewires average nine fit/control combinations. KL is native-to-reconstructed output KL averaged on 32 natural contexts; lower is better.

Agreement reconstruction improvements from H730 partly translate to task-effect preservation. However, the random-initialized learned grouping's rewires retain equally strong aggregate effects. This control preserves each atom's membership degree, so the result does not establish an advantage from the particular learned group memberships beyond those degrees. Correlation-initialized groups do retain a modest advantage over their rewires, yet remain below scalar features and have worse mean output KL. These are small reused-panel comparisons, not precise causal attribution of the training improvement to degree changes.

Individual-effect relative RMS errors for agreement are 0.0935 learned-correlation and 0.0786 learned-random, versus 0.0790 scalar; their rewires give 0.1142 and 0.0868. Thus the learned-random candidate's aggregate retention does not imply uniformly more accurate effects. For capitals, scalar error remains smaller (0.3421 versus 0.4756/0.4932). Learned-random capital fits produce two sign reversals across 48 transfer interventions; learned-correlation fits produce none. Task-minus-absolute-nuisance specificity is 4.433/4.532 for learned agreement groups versus scalar 4.502, and 1.687/1.553 for learned capital groups versus scalar 2.102.

Observed next-token CE changes remain a separate metric: +0.00601/+0.00501 agreement and +0.00798/+0.01738 capitals for learned-correlation/learned-random groups; scalar changes are +0.00194/+0.00707. Reconstruction, retained task effects, output KL and CE do not provide an interchangeable ranking.

Audits: 5,472 full-model example forwards completed. All input/source hashes remained unchanged; KL values were finite. Identity task replay maximum error was 3.81e-6, identity native reconstruction KL 3.64e-12, and scalar task effects exactly reproduced H725. H730's four implementation tests passed. No new natural documents or original reserved-test evaluations were used.

Decision: stop tuning this support-energy objective for now. The alternatives are implemented and diagnostically tested, but the present grouped models are not verified manifolds or a general improvement over scalar featurizers. Preserve all seeds and both learned initializations; do not select a winner from this panel. Next use fresh prompts across agreement, factual recall, entity binding, sentiment and sequence copying, with native-competence and exact-head intervention controls. Evaluate frozen scalar and group candidates, and require selective shared-component effects beyond degree-matched controls before assigning semantic or computational meaning. The broader manifold, hierarchy and DAG questions remain open.
