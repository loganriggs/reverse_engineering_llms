# A trained three-rule toy separates causal coverage from semantic grouping

The [546M-head multi-group red team](MULTIPLE_RANK96_GROUPS_RESULTS.md) found one robust high-energy group but no reliable full partition from energy ranking or a simple co-activation graph. To validate a different grouping rule, this test uses **ten previously trained three-head count-language models** with known A/B/C count-comparison rules, a rare query that rotates the outputs, and a rare whole-combination holdout. It is a controlled target for asking whether a six-dimensional, intervention-aware key space can be divided into complete variable groups without using labels during fitting or grouping.

From each model's weights, construct exact query × key × output tables for the original attention and three QK-branch edits: QK1 score set to one, QK2 score set to one, and both scores set to one. Original token-table logits match direct model logits on fresh sentences to maximum absolute error **2.84 × 10⁻¹⁴**. The ordinary and single-edit tables are independently Frobenius-normalized, stacked, and truncated to a six-dimensional shared key space by SVD. The **double edit is withheld** from that fit. A second condition includes all four tables. In each fixed subspace, a key-dictionary L1 rotation chooses a sparse gauge without task labels. SVD and 32 random orthogonal gauges per model are controls. The rotation preserves each subspace projection exactly.

## Counterfactual coverage has a cost

Mean relative table MSE over the ten trained models:

| Six-dimensional fit | Original | QK1 constant | QK2 constant | Both constant |
|---|---:|---:|---:|---:|
| Original + two single edits | **.00056** | **.00326** | **.00230** | .05047, withheld |
| All four conditions | .00237 | .00600 | .00493 | **.00656** |

Thus, coverage of the double edit is not guaranteed by fitting the two single edits, even at a rank that reconstructs each fitted condition well. Adding it improves that counterfactual substantially while trading away a little fidelity on the other three. These are *function-table* errors, not next-token KL or a proof of physical circuit recovery.

## Which unlabeled grouping signature works?

The sparse six-axis dictionary often represents one A/B/C contrast using two token-specific axes. A group is declared **complete** post-fit only if three *distinct* graph groups each have both >.9 coverage of one unit key-token contrast and >.9 original-function contrast-effect purity. These known contrasts are used only for evaluation. Connected components determine group number and size from three alternative similarity matrices:

1. **Ordinary-function signature:** absolute cosine between each atom's query/output response in the original function.
2. **Stacked causal signature:** the same cosine after stacking the fitted original and branch-edited query/output responses.
3. **Sentence co-activity:** absolute Pearson correlation of atom coordinates summed over 8,192 unlabeled training-distribution sentences; the rare whole-combination holdout is excluded.

| Grouping rule at cosine/correlation threshold .8 | SVD, seeds with complete A/B/C | Sparse gauge, seeds with complete A/B/C | Random gauge, successful rotations |
|---|---:|---:|---:|
| Original-function response | 0/10 | **10/10** | 0/320 with single-edit fit; 2/320 with all-four fit |
| Stacked causal response | 0/10 | 0/10 | 0/320 |
| Sentence co-activity | 0/10 | 0/10 | 0/320 |

The sparse ordinary-function graph finds **three groups of two axes in all ten seeds** at threshold .8, for either fit condition. It passes the same completeness criterion in all ten seeds at each predeclared threshold .6, .7, .8, and .9; one seed's .9 graph has sizes 2/3/1 rather than exactly 2/2/2. For the single-edit fit, mean matched contrast coverage is **.9993** and mean original-function effect purity **.9998**. The all-four fit also passes 10/10 at all four thresholds.

The stacked-signature failure needs careful interpretation. The key space is obtained by SVD of precisely that stack, so its unrotated columns are orthogonal by construction; ordinary column cosine is a poor grouping signal for that object. After sparse rotation, branch edits also distinguish the two token-specific axes that implement the same *ordinary* bit. This experiment rejects this **naive stacked cosine**, not causal information as a class. The co-activity failure is consistent with variable token counts and independent count sources: two axes for the same rule need not fire in the same sentence.

## Held-out task deletion

At threshold .8, remove each post-fit ordinary-function group from the six-dimensional **replacement token function**, leaving the trained readout and other two groups in place. With the single-edit fit, the reduced model reaches **100% joint and rare-heldout accuracy** in every seed on independent 10,000-sentence tests at Poisson count rates .35, .75, and 1.2. The all-four fit also reaches 100% rare-heldout accuracy in every seed; one seed scores 99.96% full-test joint accuracy at rate 1.2. After group deletion under the single-edit fit, mean accuracy on the intended bit is **.500, .499, and .500** respectively, while mean accuracy on each unaffected bit remains **1.000** at all three rates. The grouping rule never saw these targets or the rare withheld combination. This is selective causal validation of the *reduced explanatory function*, not evidence that each group is a separately localized physical attention head.

On the withheld double QK edit, the original-function groups' matched contrast-effect purity remains high on average (.959 for the single-edit fit), but one seed/bit drops to .231; the all-four fit averages .953 with a different outlier at .105. Semantic-looking ordinary groups therefore do not guarantee uniformly selective branch-edited effects. The table-fidelity gate is essential for any physical-intervention claim.

## Implication for the 546M head

This toy supports a **two-stage canonicalization**: first choose a shared subspace for the natural and counterfactual functions one wants to preserve, then group its sparse axes using the *ordinary function's* response signature, with counterfactual fidelity and selective effects tested separately. It also explains why concatenating every intervention into one similarity vector can erase a semantic grouping even while improving causal coverage. The next real-model test should construct ordinary query/output response signatures for the rank-96 QK1 components across held-out positions and token panels, discover groups without a supplied count, and ask whether those groups predict selective full-model context effects. The trained toy's six token directions and clean three-rule grammar make this an existence proof for the method, not evidence of semantic groups in natural language.

Reproduce with `three_count_causal_grouping.py` and `three_count_causal_grouping.py --include-double`. The ten existing `three_count_seed*.pt` checkpoints and new `three_count_causal_grouping.json` and `three_count_causal_grouping_all_four.json` retain exact gates, per-seed dictionaries, group assignments, reconstruction errors, random controls, and rate-shift deletion results.
