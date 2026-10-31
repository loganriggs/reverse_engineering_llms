# Same edit category is not enough for bracket-axis transfer

The [answer-free intervention gauge](LATER_HEAD_UNLABELED_INTERVENTION_RESULTS.md)
used two independent 800-edit panels and found stable rank-four axes in
selected bracket and agreement heads. The panels contain no identical
**full prompt pairs**, but they repeat token-ID substitutions: all 80
bracket edits in each panel exchange the same `Ġ(`/`Ġ[` token pair;
the noun-number edits cover ten singular/plural token pairs. The prior
lexical and format tests did not establish transfer across new bracket
token identities.

We froze the rank-four weight cuts and raw-PCA answer-free axes fitted
independently on the two panels. A predeclared new panel has 18
brace→parenthesis or brace→square prompts (two unseen token-ID swap
pairs) and 12 singular/plural prompts using **dog, student, child,
horse** (four unseen pairs). Every swap is absent from both fitting
panels. No new prompt or answer was used to refit or select an axis.
We measured native answer competence and finite native-head, rank-four,
and single-axis paths.

| Holdout | Native answer pairs correct | Rank-four vs native head-path error | Frozen axis vs rank-four path error |
|---|---:|---:|---:|
| Brace→parenthesis | 0/9 | .006 | **.877** |
| Brace→square | 0/9 | .027 | .161 |
| New singular/plural nouns | **12/12** | .056 | **.032** |

Errors are mean absolute path differences divided by mean absolute
reference path, using the axis fitted on the first intervention panel.
The independently fitted second-panel axis gives nearly the same
overall results: .452 relative error across all brace swaps and .026
on new nouns, versus .451 and .032 for the first panel. Across all 18
brace swaps the rank-four cut remains faithful to the native head path
(.019 relative error), but the single axis loses .451 relative to
rank four. On brace→parenthesis it retains about 12% of rank-four path
magnitude; on brace→square it retains about 84%.

The brace prompts are a **negative competence control**: the model
does not prefer the expected closing brace in any of the 18 pairs. We
cannot call the missed axis an abstract bracket-task failure. It does
show that a gauge learned from repeated parenthesis/square substitutions
does not generally compress the head's response to a new bracket token.
The agreement result is stronger: the model is competent on the new
noun pairs and the frozen axis transfers. The asymmetry is consistent
with coverage—ten noun-number substitution identities versus one
bracket identity during fitting—but does not prove coverage is the
sole cause.

Stability across contexts that reuse one token pair is not stability
across the input variable's full domain. A next bracket test needs a
diverse edit-identity training measure and competent held-out bracket
forms; otherwise the axis should be described as a reliable coordinate
for the observed parenthesis/square contrast. Whole-task discovery
remains a separate gate.

Reproduce with `flagship_unseen_token_edit_transfer.py`; all prompts,
token IDs, competence checks, and finite paths are in
`full_width_results/unseen_token_edit_transfer.json`.

A [competent identity follow-up](COMPETENT_BRACKET_IDENTITY_TRANSFER_RESULTS.md)
tests six new token-ID pairs that still encode the parenthesis-versus-square
relation. All six pass the forced-choice discovery screen and independent
validation; the same frozen axis retains the rank-four path within 4.4%
pooled on validation. This narrows the brace failure to a broader relation
change or task-competence limit, rather than token-ID change alone.
