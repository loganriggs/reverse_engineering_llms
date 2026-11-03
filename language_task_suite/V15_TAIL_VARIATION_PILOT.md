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
is intended to verify which tokens drive this gap on an independent
FineWeb offset before changing the grammar. It saves hashes and aggregate
counts only. The first attempt ended with a process exit 139 before
writing a result; its outcome remains pending.
