# The “as” group responds to context, but is not uniquely sensitive downstream

The [pre-RoPE group analysis](PREROPE_GAUGE_AND_GROUP_RESULTS.md) found a
reproducible response subspace in layer-0/head-0 QK1 of the 546M model.
Later [conditional tests](PREROPE_CONDITIONAL_LEXICAL_RESULTS.md) narrowed
its apparent connective association mostly to the key IDs for `" as"`
and `" so"`. This experiment asks whether the group carries a reusable
construction-specific computation, or whether a lexical response and
generic causal-position effects explain the observation.

## A fixed-token construction panel

We generated 72 new, short prompts across nine predeclared strata:
first and second `" as"` in a comparison, `" as"` as a role marker, in
a clause, first and second in `" as well as"`, and `" so"` as degree,
result, or purpose. Each stratum has eight unique prompts. The edited
key token is always tokenizer ID 355 (`" as"`) or 523 (`" so"`), so
contrasts among roles of the same token do not rely on different key IDs.
The fixed pre-RoPE rank-96 cut and response group predate these prompts.

Deleting the response group at full strength gives very different mean
KL after the selected key, but also removes very different amounts of
head-response energy:

| Construction | Mean after-key KL | Mean removed pre-projection head energy |
|---|---:|---:|
| Comparison, first `as` | .00491 | 609 |
| Comparison, second `as` | .0000085 | 17 |
| Role `as` | .000103 | 196 |
| Clause `as` | .0000028 | 17 |
| `as well as`, first `as` | .00844 | 1,054 |
| `as well as`, second `as` | .0000238 | 64 |
| Degree `so` | .000135 | 195 |
| Result `so` | .0000609 | 81 |
| Purpose `so` | .0000325 | 84 |

This demonstrates context-sensitive activation of the same token ID,
but the raw KL differences do not identify a special downstream channel:
intervention dose varies by orders of magnitude. A rank-ten Haar group
shows the same qualitative first-versus-second `as` pattern at full dose.

## Pairing the two occurrences and matching dose

We then made 24 **fresh paired sentences**: twelve comparisons and twelve
`" as well as"` constructions, each with exactly two occurrences of
token ID 355. For each sentence, we edited the first or second `as`
separately and scored **only the shared suffix beginning at the second
`as`**. This removes sentence content and evaluation-position differences
from the paired comparison.

First, within each group and prompt, we set both deletion fractions to
remove 25% of the smaller full-dose *pre-projection* head energy.
The first `as` still caused about 10.5× the second's common-suffix KL
in comparisons and 1.6× in `" as well as"`. But three predeclared
Haar rank-ten groups showed the same directional pattern: 8.3–14.6×
for comparisons and 1.1–1.4× for `" as well as"`.

The learned `c_proj` block has anisotropic gain (head-0 Gram condition
number about 20), so pre-projection energy is an imperfect dose match.
We repeated the assay matching **post-`c_proj` residual energy** across
*all four groups and both occurrences within each sentence*. The target
was the minimum of the eight full-dose energies on that sentence; all
fractions were at most one. The maximum relative mismatch of achieved
post-projection energy was 2.3 × 10⁻⁷.

| Construction and occurrence | Discovered group common-suffix KL | Three Haar-group means, range |
|---|---:|---:|
| Comparison, first `as` | 5.27e-6 | 3.43e-6–5.73e-6 |
| Comparison, second `as` | 0.400e-6 | 0.335e-6–0.383e-6 |
| `as well as`, first `as` | 4.01e-6 | 3.46e-6–4.27e-6 |
| `as well as`, second `as` | 2.99e-6 | 2.83e-6–2.99e-6 |

The discovered group's mean is inside the random-group range in three
rows and 4% above its upper edge for the comparison's second `as`.
Comparison first/second KL ratios are 13.2 for the
discovered group and 9.3–15.0 for the Haar controls; `" as well as"`
ratios are 1.3 for the discovered group and 1.2–1.4 for controls.
Thus the first-versus-second difference persists when the exact token,
sentence, suffix, and residual dose are controlled, but it is **not
specific to this discovered subspace**. It may reflect generic causal
position, construction, or attention-architecture effects. This test
does not distinguish those explanations.

These are curated prompts, not a natural-document sample, and KL is
not an answer-level competence measure. The result still supports the
earlier narrow finding that this group activates strongly in some
`as` contexts. It does **not** support interpreting it as a uniquely
downstream-sensitive comparison or coordination module. We stop this
single-lexeme branch here. The broader canonicalization program should
return to multiple learned rules and whole-task holdouts, using matched
post-projection dose when proposing causal groups.

Reproduce with `flagship_prerope_construction_strata.py`,
`flagship_prerope_paired_as_dose.py` (four fixed groups), and
`flagship_prerope_cross_group_dose.py`. The prompt texts, target
occurrences, doses, energy gates, and per-prompt KL are saved in the
matching `full_width_results/prerope_*` JSON files.
