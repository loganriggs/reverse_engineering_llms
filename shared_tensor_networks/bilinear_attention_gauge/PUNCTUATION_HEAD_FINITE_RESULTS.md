# Punctuation-response heads have a small downstream path

The [answer-free 36-head scan](ALL_HEAD_UNLABELED_EDIT_SCAN_RESULTS.md)
ranked layer-13 heads 3 and 5 highly and found that their leading
edit-response axes were dominated by punctuation substitutions. We froze
those axes from the training intervention panel and tested 20 independent
validation edits each of punctuation, numbers, and brackets. At the
final token, we transplanted either the complete native head-output
difference or its leading-axis component into the unedited prompt, then
measured the resulting **full-vocabulary logit-vector change**. Every
head in layer 13 was evaluated. The edit's unpatched logit-vector change
served as the reference; the signed projection of the patched change
onto it measures how much of that change follows the head path. This is
a sufficiency-style path measurement, not a decomposition of the total
edit or a necessity test.

| Edit / head | Complete head: mean signed projection | Leading axis | Complete-head mean cosine with edit |
|---|---:|---:|---:|
| Punctuation / 13.3 | .053 | .027 | .262 |
| Punctuation / 13.5 | .038 | .021 | .312 |
| Punctuation / 13.8 control | .014 | -.001 | .105 |
| Number / 13.3 | .000 | .000 | -.009 |
| Bracket / 13.8 known locus | .311 | .313 | .760 |

Nonparametric case-bootstrap 95% intervals for the punctuation full-head
projections are [.045, .060] for head 3 and [.028, .049] for head 5;
their leading axes yield [.018, .036] and [.012, .031]. The bracket-head
control interval on bracket edits is [.297, .327]. Thus the punctuation
heads have small, selective downstream effects on this panel, but their
large input-side edit responses do **not** translate into a comparably
large output path. The leading axis also retains only about half of the
punctuation path, unlike the bracket head where one axis retains the
complete-head projection.

These tests use new token pairs but the same designed edit categories and
prompt templates as the scan. They do not establish a punctuation task,
rule recovery, generalization to new formats, or redundancy among heads.
Further punctuation-specific interpretation should require a frozen
output prediction on new formats or a stronger finite causal effect.
For the broader canonicalization comparison, this is a useful negative
control: **response sparsity can select a crisp input variable without
selecting a strong downstream mechanism**. The toy metric-by-prior
experiment can now test this failure mode with known ground truth.

Reproduce with `flagship_punctuation_head_finite.py`; the 540
head-by-case records and summary are in
`full_width_results/punctuation_head_finite.json`.
