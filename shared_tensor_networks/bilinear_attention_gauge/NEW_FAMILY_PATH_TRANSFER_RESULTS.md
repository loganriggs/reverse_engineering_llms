# New edit families expose a low-rank transfer gap

The [training-only floor rule](PATH_FLOOR_GUARDRAIL_RESULTS.md) chose a
mixture that improved familiar selected edit families while mostly
protecting familiar unselected families in the **immediate** head output.
We tested whether that advantage transfers to genuinely new *edit-family
types* in the 546M model.

For L9H7 we predeclared four unseen token-substitution families: modal
verbs (`can` versus `must`/`might`/`will`), logical connectives (`and`
versus `or`), auxiliary tense (`was` versus `is`), and code comparisons
(`==` versus `!=`). Each has 12 newly generated, unique, one-token edit
pairs. Their exact pairs and family labels were excluded from the prior
training and validation edit panels; none entered the native-path
top-three selector or the floor rule. The edited text is short and
templated, so this is a **whole-designed-family holdout**, not natural
corpus transfer or blind task discovery.

At each rank we compare four orthogonal head-state subspaces fitted
without the new families: weight-only `c_proj` Gram; pooled training-edit
PCA; the fixed .1 selected-family floor; and the training-only guardrail
floor. The guardrail selects .5, .35, and .2 at ranks 32, 64, and 96. We capture
the new edit's native head-state difference from the full model, patch it
into the base prompt, and compare the final-logit patch path with the
path obtained after projecting the base head state and edit difference
into each reduced subspace. The path metric is per-case squared error
normalized by the native path squared norm. We separately measure
forward KL on ordinary base prompts.

| Rank | Fit | Mean new-family path MSE | Mean base-prompt forward KL |
|---:|---|---:|---:|
| 32 | Weight-only | .392 | .01278 |
| 32 | Pooled edits | .428 | .00459 |
| 32 | Fixed .1 floor | .433 | .00501 |
| 32 | Training guardrail | .437 | .00450 |
| 64 | Weight-only | .181 | .00673 |
| 64 | Pooled edits | .184 | .00164 |
| 64 | Fixed .1 floor | .186 | .00242 |
| 64 | Training guardrail | .183 | .00216 |
| 96 | Weight-only | .066 | .00201 |
| 96 | Pooled edits | .068 | .00075 |
| 96 | Fixed .1 floor | .070 | .00070 |
| 96 | Training guardrail | .069 | .00067 |

The path-selected measures provide **no overall new-family transfer
advantage** at ranks 32, 64, or 96. Increasing rank substantially improves all
four methods, which is evidence for a capacity/coverage component. The
weight-only cut slightly favors new-family path fidelity at these ranks,
but has markedly worse ordinary forward KL; one should not call it a
globally better canonical reduction. Native median path norms on the new
families are nonzero and sizable (modal 13.0, connective 5.6, tense 15.7,
code comparison 12.8 in final-logit L2 units), though path magnitude by
itself does not establish contribution to a meaningful answer.

Rank-32 per-family path MSE for pooled versus training-guardrail is:
modal .469/.499; connective .499/.449; tense .313/.351; code comparison
.432/.447. The rule can help one new family while hurting three others.
At rank 64 the same comparison is .260/.252; .190/.203; .109/.097;
.177/.181. This is not a stable transferable advantage.

The result limits the stage-one decision: local fidelity on a fixed
taxonomy of edit types does not certify broader causal fidelity. The
next experiment should enforce a training-only **absolute** downstream
path or KL criterion, then challenge it on held-out edit families and
natural documents. A sparsity or reusable-component prior should be
compared only after the reduced function passes that declared gate.

Reproduce with `flagship_new_family_path_transfer.py`, setting
`ATTN_RANK` to 32, 64, or 96. Saved cases and measurements are in
`full_width_results/new_family_path_transfer_rank*.json`.
