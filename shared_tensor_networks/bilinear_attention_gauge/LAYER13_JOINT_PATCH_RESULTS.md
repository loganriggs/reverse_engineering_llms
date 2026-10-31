# Punctuation paths at layer 13 add, but explain little of the total edit

The prior [single-head finite test](PUNCTUATION_HEAD_FINITE_RESULTS.md)
found selective yet small logit effects from two punctuation-responsive
heads. To check whether the limitation was head-level distribution or
cancellation, we patched native edit differences into heads 3 and 5
together, then all nine layer-13 heads. The 20 independent validation
edits of each type and full-vocabulary logit readout are unchanged.

The signed projection below is the patched logit change projected onto
the unpatched input edit's logit change, divided by the squared norm of
that total edit change. A value of .09 means a same-direction path with
roughly 9% of the total change in this sense; it is not an explained-
variance fraction or necessity measure.

| Edit | Head 3 | Head 5 | Heads 3+5 | All 9 heads | Head 8 |
|---|---:|---:|---:|---:|---:|
| Punctuation | .053 | .038 | **.090** | **.116** | .014 |
| Number | .000 | .004 | .004 | .029 | .000 |
| Bracket | .005 | .005 | .010 | .326 | **.311** |

For punctuation, the joint effect of heads 3 and 5 is almost the sum of
the individual effects: the average norm of the joint-minus-sum logit
vector is only **0.4%** of the total punctuation edit's logit-change norm.
Thus there is little evidence of cancellation between these two paths at
this intervention dose. They account for most of the same-direction
layer-13 head contribution: all nine heads reach only .116. The small
downstream fraction comes mainly from other routes of the input edit,
not hidden cancellation among the two candidate heads. For brackets,
head 8 alone contributes .311 versus .326 for all nine heads, reinforcing
the contrast between a dominant head path and a distributed, small one.

This is a final-position transplantation at one layer. It does not
decompose earlier-layer changes, prove necessity, or assign a semantic
output rule to the punctuation heads. The next canonicalization test
should compare priors on a known-rule toy that includes both a dominant
causal head and a head with a large, legible input response but little
downstream impact; for the real model, do not spend a long single-head
interpretation effort on punctuation until a stronger output prediction
is found.

Reproduce with `flagship_layer13_joint_patch.py`. Per-case and aggregate
results are in `full_width_results/layer13_joint_patch.json`.
