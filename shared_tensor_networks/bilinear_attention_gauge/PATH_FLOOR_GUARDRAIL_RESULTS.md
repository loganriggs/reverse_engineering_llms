# A training-only coverage floor helps locally, but does not fully protect downstream paths

The [path-selected reduction experiment](ALL_HEAD_PATH_SELECTED_REDUCTION_RESULTS.md)
improved the edit families with the largest native downstream path norms, at
the cost of unselected families. We tested a training-only rule for choosing
the mixture between those three selected families and all ten designed edit
families, separately for each of 36 later-layer heads and each rank.

At ranks 16, 32, 64, and 96, we fit spectral projectors using eight fixed
floors (0, .05, .1, .2, .35, .5, .75, 1), plus pooled-edit PCA. A head's
three target families come from its **training** native full-model path
magnitudes, without answer labels. On training responses, the rule chooses
the candidate with the smallest mean selected-family immediate-`c_proj`
relative MSE subject to worst unselected-family MSE no more than **.02
absolute** above pooled-edit PCA. Because pooled PCA itself is a candidate,
the rule always has a feasible fallback. No validation edit selects the
floor or rank.

| Rank | Selected-family error, rule / pooled | Worst unselected-family error, rule / pooled | Aggregate error, rule / pooled | Heads where rule improves selected error | Heads exceeding .02 unselected margin on validation |
|---:|---:|---:|---:|---:|---:|
| 16 | .289 / .331 | .739 / .731 | .387 / .360 | 33/36 | 10/36 |
| 32 | .171 / .204 | .534 / .532 | .244 / .223 | 35/36 | 7/36 |
| 64 | .074 / .087 | .259 / .258 | .106 / .095 | 34/36 | 6/36 |
| 96 | .028 / .032 | .100 / .091 | .040 / .034 | 32/36 | 5/36 |

All table errors use **independent same-type validation edits** and are
means across heads. Selected and worst-unselected errors normalize each
family separately before averaging; aggregate error pools the edited panel.
The training guardrail transfers reasonably on average, but it is not a
per-head validation guarantee. The 10, 7, 6, and 5 violations are especially
important because the rule's constraint was applied to a small training
panel and only to the *immediate* output.

## Finite full-model check

For broad head L9H7 at rank 32, the training rule chose a **.5 floor**.
We evaluated 12 independent validation cases each for its selected noun
number, preposition, and name families, plus an unselected bracket guardrail.
The metric is per-case relative MSE between final-logit **head-patch paths**:
the native edited-head-state patch versus the same edit projected into a
reduced-head model. We also measured forward KL of each ordinary base prompt
from native to reduced model.

| Family | Pooled path MSE | Fixed .1 floor | Training-rule .5 floor |
|---|---:|---:|---:|
| Noun number | .114 | **.097** | .110 |
| Preposition | .231 | **.186** | .211 |
| Name | .236 | **.200** | .218 |
| Bracket | **.122** | .372 | .193 |

Across the 36 selected-family cases, the path MSEs are .194 pooled, .161
with the fixed .1 floor, and .180 with the training-rule .5 floor. Across
all 48 base prompts, ordinary forward KL is .00235 pooled, .00290 at .1,
and .00240 with the rule. The .5 floor curbs most of the bracket-path
damage caused by .1, but bracket remains worse than pooled by .071 mean
relative MSE (11 of 12 cases). The selected-family gain is correspondingly
smaller. The full-model downstream result therefore does **not** establish
that a local `c_proj` error guardrail is sufficient.

This comparison is exploratory: it uses one rank, one head, and new cases
from the same designed edit families. It has not tested whole-family
holdouts, independent documents, task predictions, or human-legibility
priors. Native path *magnitude* was the selection signal; it may select a
large path unrelated to a useful task effect. A stronger rule should use
training-only finite-path fidelity or KL constraints, enforce an absolute
coverage threshold rather than a pooled-relative one, and then face fresh
edit families. Only a cut that passes those checks should proceed to
sparsity or reusable-component optimization.

Reproduce the all-head local rule with `flagship_path_floor_guardrail.py`
and the finite check with `flagship_path_selected_finite.py`. Data are in
`full_width_results/path_floor_guardrail.json` and
`full_width_results/path_selected_finite.json`.
