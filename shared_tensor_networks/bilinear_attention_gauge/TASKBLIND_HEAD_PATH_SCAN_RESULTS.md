# Diverse finite-path head selection: coverage helps, unseen rules remain hidden

We tested whether heads can be selected by downstream effect without using
answer labels. The fixed edit distribution contains ten deliberately balanced
one-token edit types, including bracket, noun-number, adjective, color,
country, name, number, punctuation, preposition, and random-context edits.
For each type we used 12 training and 12 independent validation edits from
the existing 800-edit panels. At the final position, each edited-minus-base
native 128-dimensional head state was patched into the base run, one head at a
time, for all nine heads of layers 9, 11, 13, and 15 (36 heads total).
Each patched full-vocabulary logit change was compared with the model's
complete logit change under the input edit. This is answer-free selection,
**not** a whole-task-blind or naturally sampled edit distribution.

For each edit and head we use the signed projection
`<patched logits - base logits, edited logits - base logits> /
||edited logits - base logits||²`. It measures alignment and amplitude
along the observed output change. Because separate patches can interact,
these scores are not additive shares of a causal mechanism; a negative
score can reflect opposition, and a high score does not identify a rule.
The batched nine-head patcher agrees with an earlier sequential patcher
on 324 overlapping cases within `3.51e-6` signed-projection units.

## Mean effect versus coverage

The mean signed-projection ranking is highly reproducible across the two
same-type panels (Spearman `0.979` over 36 heads; identical top-eight set).
Its leading validation heads are L9H7 (`0.0716`), L11H6 (`0.0513`),
L9H8 (`0.0448`), and the known bracket head L13H8 (`0.0346`).
The former punctuation input-response heads L13H3 and L13H5 rank only
13th and 17th by downstream alignment. This confirms that a large
activation response is not necessarily a large output path.

We also greedily select a *set* of heads that maximizes the mean, across
training edits, of the best nonnegative signed projection attained by any
selected head. This rewards covering distinct edits rather than repeatedly
choosing broad responders. No edit type or answer label enters the selector.

| Three-head selection rule | Selected heads | Validation mean best positive projection |
|---|---|---:|
| Mean individual downstream path | L9H7, L11H6, L9H8 | `0.0967` |
| Greedy path coverage | L9H7, L13H8, L9H8 | **`0.1194`** |
| Input head-state response energy | L13H8, L13H3, L11H5 | `0.0447` |

For the same ten edit types, a stratified validation-case bootstrap gives
greedy minus mean-path `+0.0227` (95% interval `+0.0190` to `+0.0264`)
and greedy minus input-energy `+0.0747` (`+0.0672` to `+0.0831`).
The intervals quantify case sampling **conditional on these designed edit
types**, not generalization to new tasks. In 2,000 stratified resamples of
the training cases, L9H7 and L13H8 enter the greedy top three every time;
L9H8 enters 80.1%, and the exact top-three set recurs 80.1%.

The coverage gain has an identifiable tradeoff. On bracket validation edits,
the greedy top three score `0.322` versus `0.023` for mean-path top three;
on adjective edits they score `0.104` versus `0.122`. At five heads both
selectors score `0.131` overall. The selector gets a bracket specialist into
a small budget, rather than producing a universal improvement on all edits.

## Whole-type exclusion is the discovery gate

When all bracket edits are withheld from fitting, L13H8 falls from fourth
to **24th** in the mean-path ranking and disappears from the greedy top five.
The latter instead selects L9H7, L9H8, L11H6, L13H0, L11H3.
Thus the method can find a specialist only when its response is excited
by the unlabeled fit distribution. It has not discovered a hidden bracket
rule from unrelated examples. The validation panel repeats the training
edit families, so even the positive coverage result is transfer across
examples, not across tasks.

The next informative test is a trained toy language model with known
multiple rules, nuisance token changes, and a genuinely withheld rule or
composition. Fit reduction using both weight-only broad token coverage and
natural usage, then compare exact-gauge legibility objectives using
unlabeled edit responses and downstream paths. Predeclare a selector before
the withheld task is revealed; evaluate its finite interventions, task
prediction, and matched random groups separately. On the 546M model, the
same gate requires new edit families or genuinely new output predictions,
not another split of the ten designed types.

Reproduce the finite scan with `flagship_taskblind_head_path_scan.py` and
the selection analysis with `analyze_taskblind_head_path_scan.py`. Complete
per-edit scores, rankings, bootstrap intervals, and whole-type exclusions
are in `full_width_results/taskblind_head_path_scan.json` and
`full_width_results/taskblind_head_path_scan_analysis.json`.
