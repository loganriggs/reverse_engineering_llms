# A known bracket group transfers to nested closers, with limits

The [joint bracket reduction](JOINT_HEAD_MOMENT_FRONTIER_RESULTS.md)
and [answer-free two-axis group](JOINT_BRACKET_LEGIBILITY_RESULTS.md)
were fitted from simple parenthesis-versus-square edits. To challenge
their interpretation, we tested **nested** delimiters. An *inner*
pair changes the most recent opener, `([expr` versus `((expr`, so the
expected next closer changes from `]` to `)`. An *outer* pair changes
an earlier opener after a fixed inner bracket has closed,
`([expr]` versus `[[expr]`, so the expected closer changes from `)`
to `]`. This tests whether the head path survives a change in which
opening bracket matters, rather than only a new arithmetic expression.

The exploratory panel contains three prefixes and four expressions per
relation: **12 inner and 12 outer pairs**, all included in the report.
Each pair changes exactly one token ID, and none of those token swaps
entered the original 80-edit group fit. The native model passes the
two-closer forced-choice gate on **12/12 inner** and **10/12 outer**
pairs. The two failures are both outer prompts beginning `Review`.
This is a weak generative competence test: both expected closers are
top-1 in only 1/12 inner and 0/12 outer pairs; the median target rank
is 2 for inner and 6.5 for outer. We therefore report forced-choice
behavior, not next-token accuracy.

We patch the edited-minus-base final-position L13 H8 head state into
the base run, comparing the complete native head with frozen
weight-only and joint-λ=.1 rank-four cuts. Inside each cut, the
answer-free two-axis edit-PCA group was fitted *before* these nested
cases. Errors below are mean absolute finite `)` minus `]` margin-path
error divided by the mean absolute relevant reference path.

| Relation | Native pair competence | Native head path magnitude | Weight rank-four error vs native | Joint rank-four error vs native | Joint one-axis error vs its cut | Joint two-axis error vs its cut |
|---|---:|---:|---:|---:|---:|---:|
| Inner closer | 12/12 | 2.07 logits | 8.25% | **2.80%** | 77.70% | **1.69%** |
| Outer closer | 10/12 | 2.14 logits | **2.56%** | 6.01% | 30.48% | **1.31%** |

The weight cut's two-axis group misses its own cut by 1.66% on inner
and 0.89% on outer pairs. The joint two-axis group has the expected
path sign on all 24 cases, including the two native forced-choice
failures. Restricting to the 22 competent pairs changes the joint
rank-four errors to 2.80% inner and 5.46% outer; its two-axis errors
against its own cut are 1.69% and 1.11%. The top edit-energy axis
alone is particularly inadequate for the *inner* relation, despite
explaining 99% of the simple-bracket training response energy. This
supports retaining a small **group**, not naming one dominant axis a
complete bracket mechanism. The different weight/joint rankings for
inner and outer also show why one selected behavior is insufficient
to certify a reduction as universally faithful.

We then removed native head-state components before the output
projection, rather than adding an edit-response patch. This is a
stronger but off-manifold intervention. The table gives correct
two-closer pairs after removal; native competence is in parentheses.

| Removed final-position subspace | Standard bracket (24/24) | Nested inner (12/12) | Nested outer (10/12) |
|---|---:|---:|---:|
| Fitted joint two-axis group | 24 | 11 | **4** |
| Complementary two axes inside joint cut | 24 | 12 | 10 |
| Full joint rank-four cut | 24 | 11 | **4** |
| Entire selected 128-dimensional head | 24 | 11 | 3 |
| Sixteen random two-axis groups | all 24 | median 7; range 3–12 | median 10; range 3–12 |

For outer closers, the fitted group removal changes the native base
margin enough to lose six previously correct pairs, while its
complement has no pair-level effect. Its absolute margin shift is
0.87 logits, versus 0.86 for removing the whole joint cut; on
standard prompts the group and whole-cut shifts are both 2.34 logits.
This is evidence that the group carries much of the selected head's
causal contribution in these contexts. It is **not** proof of a
necessary answer circuit: the two-closer choice is weaker than top-1
generation, other heads can preserve answers, and two of sixteen
random rank-two groups are at least as disruptive on the outer panel.
State subtraction also moves the model off its native activation
distribution.

This is the stopping point for the present single-head bracket detour.
It established a transferable, compact finite-response group and a
limited removal effect, but did not show unique internal computation
or newly discovered task semantics. The next canonicalization test
should select candidate heads and edits without using the bracket
label, then demand predictions on withheld behavior families. A
trained toy language model with known multiple rules remains the
cleaner place to validate that discovery criterion.

Reproduce the unfiltered nested screen and finite paths with
`flagship_nested_bracket_transfer.py`, and the removal panel with
`flagship_nested_bracket_ablation.py`. All prompts, token IDs, target
ranks, margins, paths, random controls, and summaries are in
`full_width_results/nested_bracket_transfer.json` and
`full_width_results/nested_bracket_ablation.json`.
