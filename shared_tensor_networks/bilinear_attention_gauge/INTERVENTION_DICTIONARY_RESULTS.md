# The cost of an intervention-aware key space on trained attention toys

The [physical-head red team](JOINT_FUNCTION_PHYSICAL_REDTEAM.md) found a two-direction key dictionary that reconstructs ordinary head behavior but fails QK-branch edits. This experiment fits a shared key *subspace* to different sets of exact, weight-derived token tables. It separates ordinary-function compression from the ability to describe specified internal interventions.

For each of ten trained two-head models, construct an \(8\times6\) table for the two physical heads under each condition: original, QK1 score replaced by one, QK2 score replaced by one, or **both** QK scores replaced by one. Each table stacks four query–output rows per head. Fit a rank-\(r\) right singular subspace of the selected table stack, with each condition normalized to unit Frobenius norm so a branch edit's scale does not by itself dominate. Project all four conditions onto the selected key space and measure relative table MSE. No sampled sentences or task labels enter fitting.

| Fit conditions | Key rank | Original error | QK1-constant error | QK2-constant error |
|---|---:|---:|---:|---:|
| Original only | 2 | **.00427** | .466 | .568 |
| Original + QK1 | 2 | .156 | .179 | .529 |
| Original + QK2 | 2 | .231 | .522 | .205 |
| All three | 2 | .203 | .246 | .323 |
| All three | 3 | .102 | .0928 | .0850 |
| All three | 4 | **.00903** | **.0110** | **.00734** |
| All three | 5 | .00061 | .00241 | .00194 |

Values average ten seeds. At rank two, including either branch edit trades away substantial original fidelity, and the included edit does **not** generalize to the other branch edit. A shared rank-four key space cuts mean relative MSE for **all three** conditions below .012. This is a concrete robustness–compression tradeoff: the ordinary function is effectively two-dimensional here, while preserving both tested branch counterfactuals requires roughly four directions at the one-percent error scale. Full key rank six represents every table exactly.

## A held-out *combination* of interventions

Fit on the original function and both **single** branch edits, then test the **double** edit without using it in fitting:

| Fit conditions | Rank | Original | QK1 constant | QK2 constant | Both constant, held out if absent |
|---|---:|---:|---:|---:|---:|
| Original + both single edits | 4 | .00903 | .0110 | .00734 | **.105** |
| Original + both single edits | 5 | .00061 | .00241 | .00194 | **.0400** |
| All four conditions | 4 | .0122 | .0188 | .0234 | .0209 |
| All four conditions | 5 | .00385 | .00504 | .00404 | **.00406** |

The held-out combination does not follow from low error on each single edit. Write the two QK scores as \(S_1,S_2\) and the value/output table as \(C\), with pointwise multiplication implicit. The four functions are \(S_1S_2C\), \(S_2C\), \(S_1C\), and \(C\). Their interaction contrast is

\[
C-S_2C-S_1C+S_1S_2C=(1-S_1)(1-S_2)C.
\]

This extra cross term explains why fitting original plus the two single edits need not cover the double edit. It is an exact algebraic relationship, while the reported errors quantify the learned key subspace's failure to include the corresponding direction. At rank five, including the double edit lowers its mean error from .0400 to .00406, with modest increases on the previously seen conditions. This gives a concrete reason to include **compositions of interventions** in a causal canonicalization benchmark rather than checking only one edit at a time.

This supports using intervention response as an explicit canonicalization objective when the goal is a causal description of internal computation. It also shows why an apparently meaningful ordinary-function basis is insufficient evidence for that stronger claim. The intervention conditions were supplied in advance, not discovered; the result is fidelity to known edits, not recovery of the true graph or semantic labels. A next useful test would hold out a **different strength or family** of branch perturbation, train the dictionary on a subset of interventions, and ask whether a few added directions predict those unseen responses. The present leave-one-branch-out rows show that transfer is poor for the two extreme constant-score edits.

Run `dual_intervention_dictionary.py` on CPU after the ten `dual_output_seed*.pt` checkpoints. `dual_intervention_dictionary.json` records every seed, rank one through six, singular values, and errors under all four conditions.
