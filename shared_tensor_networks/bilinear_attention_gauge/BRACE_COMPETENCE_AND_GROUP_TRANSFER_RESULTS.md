# Bracket response group transfers across token IDs; brace task does not

The earlier one-axis bracket gauge transferred to new tokenizations of the
parenthesis/square relation, but missed brace edits. A two-axis response group
fit to additional unlabeled brace edits inside the same weight-only rank-four
head cut retained all three edit types on new contexts. We now test whether
that extra axis damages or helps on **competent, unseen token identities**,
and separately screen for a brace-closing behavior that would warrant a
semantic interpretation of the brace response.

## Competence gate for braces

We evaluated the native 546M model on 19 code-like one-token edit pairs:
15 set-versus-list prefixes and four set-versus-tuple prefixes. The two
prefixes in each pair differ in one opening-token ID. We compared the next
token logits of `}` with `]` or `)` as appropriate. The native model preferred
the expected alternative closer on all 19 alternative prefixes, but `}` on
**zero of 19** brace prefixes. The median rank of `}` was 658 on set/list
and 166 on tuple/set. Thus these prompts provide no competent brace task on
which to validate a causal brace mechanism. This targeted exploratory screen
does not establish that the model never understands braces.

## Frozen group on competent parenthesis/square identities

We loaded the frozen projectors from the earlier bracket edit-span fit:
80 original parenthesis/square edits plus 12 brace edits from `Try` and
`Compute` contexts, all projected inside the **unchanged weight-derived
rank-four layer-13 head-8 subspace**. No final-answer labels, validation
token IDs, or validation contexts entered that fit. The validation panel
contains 21 parenthesis/square prompts across six new opening-token-ID
substitutions, including plain, colon, doubled, spaced, equals, and quoted
forms. All 21 native forced-choice answers favor the expected closer.

For each edit we patch the projected final-position head-output difference
into the original model at the base prompt and measure the finite change in
the `)` versus `]` logit margin. We compare each smaller projector to the
same weight-derived rank-four patch, rather than to the full input edit.

| Frozen response projector | Mean absolute path error / mean absolute rank-four path |
|---|---:|
| Original parenthesis/square one-axis | 4.39% |
| Original parenthesis/square edit covariance, rank two | **0.56%** |
| Raw edit-covariance rank one | 5.34% |
| Balanced edit-covariance rank one | 29.53% |
| Raw edit-covariance rank two | **0.59%** |
| Balanced edit-covariance rank two | **0.60%** |
| Weight-SVD rank two | 26.43% |
| Random rank-two planes, median of 32 | 55.09% |

The mean absolute rank-four margin path is 5.30 logits. The two-axis
projectors were below 1.1% relative path error in every validation format.
None of the 32 random planes beats the two edit-derived groups; the best
random plane misses .85%. Critically, the rank-two projector fit *only* on
the original parenthesis/square edits performs just as well as either
brace-augmented group on this competent holdout. On the separate six
brace-context cases, that original-only rank-two projector also misses
only .19% or .31% of the rank-four path (brace→parenthesis/square), compared
with .08% or .10% for the brace-augmented balanced projector. The second
direction was already present in the original edit data, despite carrying
only .73% of its edit energy.

The earlier competent-identity validation independently found the rank-four
path within 2.29% of the native full-head path. The second edit-derived axis
therefore improves finite-path approximation across token identities, but the
brace examples were **not necessary** to find it in this fixed rank-four cut.

This establishes **transferable input-response geometry** for a small group
inside a weight-only cut. It does not yet identify a learned abstract
bracket-closing rule or a physical two-node circuit. The group was selected
using a designed edit family, and the brace prompts fail the native task
gate. The next semantic test needs a competent rare relation or trained toy
language model whose rule and held-out compositions are known, with group
size selected without peeking at the final task tests.

Reproduce the competence screen with
`flagship_brace_competence_screen.py`, the projector fit with
`flagship_bracket_edit_span.py`, and the frozen transfer with
`flagship_bracket_group_competent_transfer.py`. Case-level logits, swaps,
projectors, and finite effects are in
`full_width_results/brace_competence_screen.json`,
`full_width_results/bracket_edit_span_rotations.npz`, and
`full_width_results/bracket_group_competent_transfer.json`.
