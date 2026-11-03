# V7 within-kind renderer variation: a partial, negative form result

The previous [per-kind audit](V7_ZIPF_BY_KIND_FEASIBILITY.md) showed that
changing V7's document-kind weights cannot remove its excess of `the`.
I added an **optional** `variant_rate` to the event-report renderer. At each
introduction it may use a fourth frame, such as `Liam owned an orange cup in a
drawer, labeled Meadow.` At each answer recap it may use a shorter frame, such as
`Owner of map now:` or `Last source for color on map:`. The default rate is
zero, so existing training distributions and results are unchanged.

The [paired audit](audit_v7_renderer_variant.py) uses the same sampled worlds
in baseline and variant arms. Each arm contributes 480 documents with at
least 100 alphabetic words, and frequencies are computed on the first 100
words. Every answer is checked by the visible-text parser. In the cross-report
arm, 40 worlds also receive relevant-source, irrelevant-source, reviewer-ship,
target-category, and section-order edits. All supported edits pass their
expected answer flip or invariance checks. The aggregate records are
[rate .25](results/v7_renderer_variant_rate025_n480.json),
[rate .5](results/v7_renderer_variant_rate05_n480.json), and
[rate 1](results/v7_renderer_variant_n480.json).

| Kind | Variant rate | Baseline `the` | Variant `the` | Baseline top-ten mass | Variant top-ten mass |
|---|---:|---:|---:|---:|---:|
| Event | .25 | 15.06% | 13.83% | 33.70% | 33.89% |
| Event | .50 | 15.06% | 12.55% | 33.69% | 34.50% |
| Event | 1.00 | 14.99% | 10.21% | 33.54% | 37.50% |
| Cross | .25 | 14.58% | 13.41% | 31.49% | 31.73% |
| Cross | .50 | 14.59% | 12.35% | 31.48% | 32.63% |
| Cross | 1.00 | 14.55% | 10.07% | 31.46% | 35.83% |

The rate-1 variant replaces one overused word with other repeated frame
words. It does **not** improve the overall word-frequency head; even its
10% `the` share remains over twice the FineWeb reference's 4.86%. More
template variants of this kind are not yet a credible route to natural form.
The next design should draw a broader distribution of clause structures and
lexical realizations, while preserving the independent answer parser. The
pilot does not establish that a model trained on the new renderer learns any
rule or that its text is naturalistic.
