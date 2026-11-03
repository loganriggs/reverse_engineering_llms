# V15 surface-variation pilot: rejected

The [V15 generator](surface_mixture_v15_tail_variants.py) changes only a
frequent ordinary-prose tail in the V14 sampled stream. After `because NAME`,
it samples four visible continuations: `had asked about it earlier`,
`wanted a written record`, `needed the OBJECT`, and `was waiting nearby`.
The existing local conditional action, earlier item facts, source/price
questions, mixed verb/object rules, and brackets remain in the same stream.
This is a surface test, **not** a new semantic task or a trained model.

The [two-panel audit](results/v15_tail_variants_n100_surface256_seed99831.json)
checks exact sampled-prefix replay, probability normalization, inherited
fact/object mixture arithmetic, supported edits, and V14 local action
edits on 100×256-token streams per seed. It also samples 256 streams per
arm for a form and task-coverage screen. The four tail branches all occur.

| First 40 words of 100-word toy crops | Seed 99831 V14 | V15 | Seed 99832 V14 | V15 |
|---|---:|---:|---:|---:|
| Repeated-word fraction | .2671 | .2706 | .2665 | .2729 |
| Repeated-content fraction, fixed heuristic | .1032 | .1034 | .1036 | .1069 |
| Conditional-action sites / 256 streams | 153 | 132 | 124 | 160 |
| Attribute-object sites / 256 streams | 442 | 417 | 432 | 499 |
| Bracket-close sites / 256 streams | 353 | 362 | 448 | 401 |

The held-out FineWeb offset-18,000 panel used in the [V14 report](V14_CONDITIONAL_AND_BINDING.md)
had repeated-word fraction .1694 and repeated-content fraction .1127 in
the first 40 words of 58 eligible crops. These samples are not paired with
the V15 toy draws, and the content heuristic is crude. Still, the result
changes the surface diagnosis: toy **content-word** repetition is not
obviously above that natural panel, whereas all-word repetition is much
higher. Rephrasing one repeated tail does not fix this and slightly worsens
both toy panels. **Do not adopt V15 or train on it.** The next DGP revision
should target repeated sentence frames and short structural words across
the whole stream, with the same exact-oracle and task-coverage checks.

The [fresh repetition-attribution script](audit_v14_repetition_attribution.py)
then sampled a disjoint FineWeb offset 19,000. One process exited 139 before
writing; a rerun completed. Across 57 jointly eligible 100-word windows,
the first 40 words had repeated-word fraction **.271 V14 / .281 V15 /
.167 FineWeb**. Repeated-content fractions were **.109/.114/.093** under
the fixed heuristic. Most of the all-word gap comes from short words:
V14 had 449 repeated stopword occurrences versus 234 in FineWeb, compared
with 170 versus 146 repeated non-stopwords. The most striking token is
`the`: 486 V14 occurrences and 429 repeats, versus 102 natural occurrences
and 58 repeats. The natural panel also has many `a`, `an`, `of`, and `to`
occurrences, absent or rare in V14's first 40 words. See the
[aggregate record](results/v14_v15_fineweb_repetition_attribution_skip19000_n64.json),
which saves hashes but no source prose. This is a robust **surface** clue,
not proof that changing articles alone will yield realistic language. It
argues for a whole-frame grammar revision with varied clause lengths and
determiners, while preserving the planted task dependencies and measuring
natural-language similarity on another untouched panel.
